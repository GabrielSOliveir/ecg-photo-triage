"""Risk engine determinístico. Recebe padrões + dados do paciente, retorna Risco final.

Princípios:
- Cada padrão tem um piso de risco base.
- Sinais vitais críticos sobrepõem (vital_floor).
- Modificadores clínicos sobem o piso (ex: ECG normal + dor torácica → mínimo INTERMEDIARIO).
- Risco final = max() de todas as fontes. Na dúvida, sobe."""

from schemas import PatientData, Risco
from rules.vital_signs import vital_floor, queixa_critica, max_risco, VitalAlert


_PADRAO_RISCO: dict[str, Risco] = {
    # Emergências
    "stemi_anterior": "EMERGENCIA",
    "stemi_inferior": "EMERGENCIA",
    "stemi_lateral": "EMERGENCIA",
    "supra_st_inespecifico": "ALTO",
    "isquemia_posterior_provavel": "ALTO",
    "taquicardia_ventricular": "EMERGENCIA",
    "fibrilacao_ventricular": "EMERGENCIA",
    "assistolia": "EMERGENCIA",
    "bav_3g": "EMERGENCIA",
    "bav_total": "EMERGENCIA",
    "bav_2g_mob2": "ALTO",
    "sgarbossa_positivo": "EMERGENCIA",
    "de_winter": "EMERGENCIA",
    "wellens": "ALTO",
    "brugada": "ALTO",
    "tako_tsubo": "ALTO",
    "qt_muito_longo": "ALTO",
    "qt_longo": "INTERMEDIARIO",
    "hipercalemia_padrao": "ALTO",

    # Alto / intermediário
    "infra_st_isquemico": "ALTO",
    "inversao_onda_t_profunda": "ALTO",
    "fa_alta_resposta": "INTERMEDIARIO",
    "fibrilacao_atrial": "INTERMEDIARIO",
    "flutter_atrial": "INTERMEDIARIO",
    "taquicardia_supraventricular": "INTERMEDIARIO",
    "pre_excitacao_wpw": "INTERMEDIARIO",
    "bav_2g_mob1": "INTERMEDIARIO",
    "bav_1g": "BAIXO",
    "bre": "INTERMEDIARIO",
    "bloqueio_ramo_esquerdo": "INTERMEDIARIO",
    "brd": "BAIXO",
    "bloqueio_ramo_direito": "BAIXO",
    "bdas": "BAIXO",
    "bdpi": "BAIXO",
    "inversao_onda_t": "INTERMEDIARIO",
    "onda_t_apiculada": "INTERMEDIARIO",
    "areas_inativas": "INTERMEDIARIO",
    "hipertrofia_ve": "BAIXO",
    "hipertrofia_vd": "BAIXO",
    "bradicardia_sinusal": "BAIXO",
    "taquicardia_sinusal": "BAIXO",
    "ritmo_juncional": "INTERMEDIARIO",
    "ritmo_idioventricular": "ALTO",
    "ritmo_marcapasso": "BAIXO",
    "ritmo_atrial_ectopico": "BAIXO",
    "pericardite_padrao": "INTERMEDIARIO",
    "hipocalemia_padrao": "INTERMEDIARIO",
    "ecg_normal": "BAIXO",
}


def base_risk_from_padroes(padroes: list[str]) -> Risco:
    if not padroes:
        return "INDETERMINADO"
    riscos = [_PADRAO_RISCO.get(p, "INDETERMINADO") for p in padroes]
    return max_risco(riscos)


def compute_risk(
    padroes: list[str],
    patient: PatientData,
) -> tuple[Risco, list[VitalAlert], list[str]]:
    """Retorna (risco_final, alertas_vitais, regras_disparadas)."""
    regras: list[str] = []

    base = base_risk_from_padroes(padroes)
    regras.append(f"base_padroes={base}")

    vital_risk, alertas = vital_floor(patient)
    if vital_risk != "INDETERMINADO":
        regras.append(f"piso_vitais={vital_risk}")

    final = max_risco([base, vital_risk])

    # Modificadores clínicos
    if queixa_critica(patient.queixa):
        if final in ("BAIXO", "INDETERMINADO"):
            regras.append("queixa_critica_eleva_para_INTERMEDIARIO")
            final = "INTERMEDIARIO"
        else:
            regras.append("queixa_critica_presente")

    # "ECG normal + assintomático" pode ser BAIXO. Caso contrário, nunca.
    if "ecg_normal" in padroes and not queixa_critica(patient.queixa) and vital_risk == "INDETERMINADO":
        if final == "INDETERMINADO":
            final = "BAIXO"
            regras.append("ecg_normal_assintomatico=BAIXO")

    if final == "INDETERMINADO":
        final = "BAIXO"
        regras.append("default_BAIXO_sem_sinais")

    return final, alertas, regras
