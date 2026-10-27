"""ECGFindings → lista de padrões clínicos nomeados.

Regras explícitas, não LLM. Os padrões alimentam o risk_engine e (futuramente) o manual local."""

from schemas import ECGFindings

ANTERIOR_LEADS = {"V1", "V2", "V3", "V4"}
INFERIOR_LEADS = {"DII", "DIII", "aVF"}
LATERAL_LEADS = {"DI", "aVL", "V5", "V6"}
POSTERIOR_RECIPROCAL = {"V1", "V2", "V3"}  # infra recíproco sugere posterior


def _wall_from_leads(leads: set[str]) -> list[str]:
    walls = []
    if leads & ANTERIOR_LEADS:
        walls.append("anterior")
    if leads & INFERIOR_LEADS:
        walls.append("inferior")
    if leads & LATERAL_LEADS:
        walls.append("lateral")
    return walls


def classify(findings: ECGFindings) -> list[str]:
    """Retorna lista de padrões. Pode ter mais de um (ex: 'fa_alta_resposta', 'bre')."""
    padroes: list[str] = []

    # Ritmo
    ritmo = findings.ritmo
    fc = findings.fc_bpm or 0

    if ritmo == "sinusal":
        if fc and fc < 50:
            padroes.append("bradicardia_sinusal")
        elif fc and fc > 100:
            padroes.append("taquicardia_sinusal")
    elif ritmo == "fa":
        padroes.append("fa_alta_resposta" if fc > 110 else "fibrilacao_atrial")
    elif ritmo == "flutter_atrial":
        padroes.append("flutter_atrial")
    elif ritmo == "tv":
        padroes.append("taquicardia_ventricular")
    elif ritmo == "tsv":
        padroes.append("taquicardia_supraventricular")
    elif ritmo == "ritmo_juncional":
        padroes.append("ritmo_juncional")
    elif ritmo == "ritmo_atrial_ectopico":
        padroes.append("ritmo_atrial_ectopico")
    elif ritmo == "ritmo_idioventricular":
        padroes.append("ritmo_idioventricular")
    elif ritmo == "marcapasso":
        padroes.append("ritmo_marcapasso")
    elif ritmo == "fibrilacao_ventricular":
        padroes.append("fibrilacao_ventricular")
    elif ritmo == "assistolia":
        padroes.append("assistolia")

    # Bloqueios
    for blk in findings.bloqueios:
        padroes.append(blk)
    if "bav_3g" in findings.bloqueios:
        padroes.append("bav_total")
    if "bre" in findings.bloqueios:
        padroes.append("bloqueio_ramo_esquerdo")
    if "brd" in findings.bloqueios:
        padroes.append("bloqueio_ramo_direito")

    # Supra ST → STEMI
    if findings.st_supra.derivacoes and findings.st_supra.magnitude_mm_max >= 1.0:
        leads = set(findings.st_supra.derivacoes)
        walls = _wall_from_leads(leads)
        for w in walls:
            padroes.append(f"stemi_{w}")
        if not walls:
            padroes.append("supra_st_inespecifico")

    # Infra ST
    if findings.st_infra.derivacoes:
        leads = set(findings.st_infra.derivacoes)
        if leads & POSTERIOR_RECIPROCAL and findings.st_infra.magnitude_mm_max >= 1.0:
            padroes.append("isquemia_posterior_provavel")
        if findings.st_infra.magnitude_mm_max >= 0.5:
            padroes.append("infra_st_isquemico")

    # Onda T
    for alt in findings.onda_t_alteracoes:
        if alt.tipo == "invertida":
            padroes.append("inversao_onda_t")
        elif alt.tipo == "apiculada":
            padroes.append("onda_t_apiculada")
        elif alt.tipo == "simetrica_profunda":
            padroes.append("inversao_onda_t_profunda")

    # Áreas inativas
    if findings.areas_inativas:
        padroes.append("areas_inativas")

    # Hipertrofias / sobrecargas
    if findings.hipertrofia_ve:
        padroes.append("hipertrofia_ve")
    if findings.hipertrofia_vd:
        padroes.append("hipertrofia_vd")

    # QT
    if findings.qtc_ms and findings.qtc_ms >= 500:
        padroes.append("qt_muito_longo")
    elif findings.qtc_ms and findings.qtc_ms >= 460:
        padroes.append("qt_longo")

    # Pré-excitação
    if findings.pre_excitacao:
        padroes.append("pre_excitacao_wpw")

    # Marcapasso (flag separado do ritmo)
    if findings.marcapasso and "ritmo_marcapasso" not in padroes:
        padroes.append("ritmo_marcapasso")

    # Achados especiais
    for esp in findings.achados_especiais:
        padroes.append(esp)

    # Sem nada → normal
    if not padroes:
        padroes.append("ecg_normal")

    return padroes
