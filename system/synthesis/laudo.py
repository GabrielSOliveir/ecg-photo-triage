"""Geração de laudo.

Estratégia:
1. Template Python a partir dos findings estruturados (zero alucinação).
2. Haiku polish opcional para fluência — sem permitir adicionar achados.

Validação: o texto final passa por checagem básica de inclusão dos elementos do template."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from config import ANTHROPIC_API_KEY, LAUDO_POLISH_MODEL, FEWSHOT_LAUDO, FEWSHOT_K
from schemas import ECGFindings, PatientData
from rules.pattern_classifier import classify
from rules.risk_engine import compute_risk

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


RITMO_TEXTO = {
    "sinusal": "Ritmo sinusal",
    "fa": "Fibrilação atrial",
    "flutter_atrial": "Flutter atrial",
    "tv": "Taquicardia ventricular",
    "tsv": "Taquicardia supraventricular",
    "ritmo_juncional": "Ritmo juncional",
    "ritmo_atrial_ectopico": "Ritmo atrial ectópico",
    "ritmo_idioventricular": "Ritmo idioventricular",
    "marcapasso": "Ritmo de marcapasso",
    "fibrilacao_ventricular": "Fibrilação ventricular",
    "assistolia": "Assistolia",
    "indeterminado": "Ritmo indeterminado",
}

BLOQUEIO_TEXTO = {
    "bav_1g": "BAV de 1º grau",
    "bav_2g_mob1": "BAV de 2º grau Mobitz I",
    "bav_2g_mob2": "BAV de 2º grau Mobitz II",
    "bav_3g": "BAV total",
    "bre": "BRE",
    "brd": "BRD",
    "bdas": "BDAS",
    "bdpi": "BDPI",
}

EIXO_TEXTO = {
    "normal": "",
    "desvio_esquerdo": "desvio do eixo para a esquerda",
    "desvio_direito": "desvio do eixo para a direita",
    "extremo": "eixo extremo (no man's land)",
    "indeterminado": "",
}

ESPECIAL_TEXTO = {
    "sgarbossa_positivo": "critérios de Sgarbossa positivos",
    "wellens": "padrão de Wellens",
    "de_winter": "padrão de De Winter",
    "brugada": "padrão de Brugada",
    "s1q3t3": "padrão S1Q3T3",
    "qt_longo": "QT longo",
    "qt_curto": "QT curto",
    "tako_tsubo": "padrão de Takotsubo",
    "ritmo_juncional_acelerado": "ritmo juncional acelerado",
    "pre_excitacao_wpw": "pré-excitação (WPW)",
    "hipercalemia_padrao": "padrão de hipercalemia",
    "hipocalemia_padrao": "padrão de hipocalemia",
    "pericardite_padrao": "padrão sugestivo de pericardite",
}


def laudo_from_template(f: ECGFindings) -> str:
    """Gera laudo determinístico a partir dos findings. Cada termo rastreia ao schema."""
    if f.qualidade.score_global == "INVALIDA":
        return "ECG inválido para interpretação — qualidade de imagem inadequada."

    partes: list[str] = []

    ritmo_str = RITMO_TEXTO.get(f.ritmo, f.ritmo)
    if f.fc_bpm:
        fc_str = f"FC {f.fc_bpm} bpm" + (" (variável)" if f.fc_variavel else "")
        partes.append(f"{ritmo_str}, {fc_str}")
    else:
        partes.append(ritmo_str)

    if f.eixo and f.eixo != "normal" and f.eixo != "indeterminado":
        partes.append(EIXO_TEXTO[f.eixo])

    intervalos = []
    if f.pr_ms:
        intervalos.append(f"PR {f.pr_ms}ms")
    if f.qrs_ms:
        intervalos.append(f"QRS {f.qrs_ms}ms")
    if f.qtc_ms:
        intervalos.append(f"QTc {f.qtc_ms}ms")
    if intervalos:
        partes.append(", ".join(intervalos))

    for blk in f.bloqueios:
        partes.append(BLOQUEIO_TEXTO.get(blk, blk))

    if f.pre_excitacao:
        partes.append("pré-excitação ventricular")

    if f.st_supra.derivacoes:
        d = ", ".join(f.st_supra.derivacoes)
        mag = f.st_supra.magnitude_mm_max
        morf = f" ({f.st_supra.morfologia.replace('_', ' ')})" if f.st_supra.morfologia else ""
        partes.append(f"supra de ST em {d}{morf}, magnitude máx {mag}mm")

    if f.st_infra.derivacoes:
        d = ", ".join(f.st_infra.derivacoes)
        mag = f.st_infra.magnitude_mm_max
        partes.append(f"infra de ST em {d}, magnitude máx {mag}mm")

    for alt in f.onda_t_alteracoes:
        d = ", ".join(alt.derivacoes) if alt.derivacoes else "difuso"
        partes.append(f"onda T {alt.tipo.replace('_', ' ')} em {d}")

    if f.areas_inativas:
        partes.append(f"áreas inativas em parede {', '.join(f.areas_inativas)}")

    if f.hipertrofia_ve:
        partes.append("sobrecarga ventricular esquerda")
    if f.hipertrofia_vd:
        partes.append("sobrecarga ventricular direita")
    if f.sobrecarga_atrial_esq:
        partes.append("sobrecarga atrial esquerda")
    if f.sobrecarga_atrial_dir:
        partes.append("sobrecarga atrial direita")

    if f.marcapasso and f.ritmo != "marcapasso":
        partes.append("presença de marcapasso")

    for esp in f.achados_especiais:
        partes.append(ESPECIAL_TEXTO.get(esp, esp))

    if not partes:
        return "ECG sem alterações significativas."

    texto = partes[0]
    extras = "; ".join(partes[1:])
    if extras:
        texto = f"{texto}. {extras[0].upper()}{extras[1:]}."
    elif not texto.endswith("."):
        texto = texto + "."
    return texto


POLISH_SYSTEM = """Você refina laudo de ECG estilo prontuário de cardiologista.

ENTRADAS:
- LAUDO BRUTO: descrição determinística dos achados extraídos pela vision.
- RESUMO CLÍNICO: 1-4 frases interpretativas escritas pela vision sobre o mesmo ECG.

NÃO há outras entradas. Você NÃO tem informações do paciente — não tente inferir histórico/sintomas/labs.

CONSTRAINTS (rigorosas — falha aqui = paciente recebe protocolo errado):

1. ACHADOS ESTRUTURAIS QUE VOCÊ PODE MENCIONAR = EXATAMENTE o que aparece no LAUDO BRUTO ou no RESUMO CLÍNICO. Nada mais.
   - Se st_infra vazio nos dois → NÃO mencione "infra de ST".
   - Se onda_t vazio nos dois → NÃO mencione "inversão/achatamento de T".
   - Se sobrecarga não mencionada → NÃO mencione "sobrecarga ventricular/atrial".
   - Se areas_inativas vazias → NÃO mencione "área inativa".
   - Se nenhum padrão isquêmico nos dois → NÃO escreva "padrão isquêmico/NSTEMI/SCA/Wellens/Sgarbossa/De Winter".

2. INTERPRETAÇÃO PERMITIDA — só sobre achados que JÁ ESTÃO NO BRUTO/RESUMO:
   - Agrupar: "infra de ST e T invertida em parede inferior — padrão sugestivo de isquemia inferior" (OK se o bruto tem infra inferior + T invertida inferior).
   - Qualificar: "sugere área inativa septal", "discreta", "alteração inespecífica", "compatível com".
   - Reconhecer incerteza por qualidade: "alteração possivelmente por posição de eletrodo", "imagem com qualidade parcial em V5-V6".

3. ESTILO PRONTUÁRIO:
   - Começar com ritmo + eixo.
   - Achados clínicos descritos em ordem clínica (rim, eixo, condução, ST/T, sobrecargas, áreas inativas, especiais).
   - Intervalos (PR/QRS/QTc) só se anormais — não dominar a primeira frase com métricas.
   - Português, direto, sem markdown.

4. TAMANHO PROPORCIONAL AOS ACHADOS:
   - ECG com poucos/nenhum achado → laudo curto. Exemplo: "Sinusal, eixo preservado, intervalos dentro da normalidade. Sem alterações significativas de repolarização ou condução." (~140 chars).
   - ECG com múltiplos achados → laudo até ~400 chars descrevendo todos.
   - NÃO encha o laudo com texto quando os achados são escassos. NÃO especule para preencher.

5. PROIBIÇÕES:
   - Conduta/recomendação ("solicitar X", "monitorizar", "encaminhar") → NÃO inclua.
   - Menção a paciente/clínica/sintomas/labs → NÃO inclua (você não tem essa informação mesmo).
   - Diagnóstico definitivo de SCA/IAM/NSTEMI sem achados de ST → NÃO inclua.

Retorne SÓ o laudo final, sem prefixos tipo "Laudo:" ou explicações."""


def _user_msg(laudo_bruto: str, resumo_clinico: str) -> str:
    return (
        f"LAUDO BRUTO:\n{laudo_bruto}\n\n"
        f"RESUMO CLÍNICO (vision):\n{resumo_clinico or '(vazio)'}\n\n"
        f"Produza o laudo final:"
    )


def polish_laudo(laudo_bruto: str, resumo_clinico: str) -> str:
    if not laudo_bruto.strip():
        return laudo_bruto
    try:
        response = _get_client().messages.create(
            model=LAUDO_POLISH_MODEL,
            max_tokens=400,
            temperature=0,
            system=POLISH_SYSTEM,
            messages=[{"role": "user", "content": _user_msg(laudo_bruto, resumo_clinico)}],
        )
        polished = response.content[0].text.strip()
        return polished or laudo_bruto
    except Exception:
        return laudo_bruto


# Nota anexada ao system quando há exemplos de laudo. Reforça: copie o ESTILO,
# nunca os ACHADOS — o laudo final tem que se restringir ao bruto/resumo do
# caso ATUAL, igual à regra do CONSTRAINT 1 do POLISH_SYSTEM.
_FEWSHOT_LAUDO_NOTE = (
    "\n\nVocê verá abaixo exemplos de laudos validados pela médica que assina os "
    "ECGs deste serviço, no formato (LAUDO BRUTO + RESUMO → laudo final). "
    "IMITE O ESTILO desses exemplos — estrutura, ordem dos achados, expressões "
    "recorrentes, qualificadores. O CONTEÚDO clínico do laudo final DEVE sair "
    "EXCLUSIVAMENTE do LAUDO BRUTO e do RESUMO CLÍNICO do caso atual; não "
    "importe achados, paredes ou diagnósticos dos exemplos."
)


def polish_laudo_with_examples(laudo_bruto: str, resumo_clinico: str,
                               exemplos: list[dict]) -> str:
    """Mesma tarefa do polish, mas com exemplos da médica como turnos
    user/assistant. Cada exemplo é (bruto regenerado pelo template a partir
    dos findings do exemplo, resumo do exemplo) → laudo final dela. Cai pro
    polish simples se algo falhar."""
    if not exemplos:
        return polish_laudo(laudo_bruto, resumo_clinico)
    messages: list[dict] = []
    for ex in exemplos:
        f = ex.get("_findings")
        if f is None:
            continue
        bruto_ex = laudo_from_template(f)
        messages.append({"role": "user",
                         "content": _user_msg(bruto_ex, f.resumo_clinico)})
        messages.append({"role": "assistant", "content": ex["laudo"]})
    if not messages:
        return polish_laudo(laudo_bruto, resumo_clinico)
    messages.append({"role": "user", "content": _user_msg(laudo_bruto, resumo_clinico)})
    try:
        response = _get_client().messages.create(
            model=LAUDO_POLISH_MODEL,
            max_tokens=400,
            temperature=0,
            system=POLISH_SYSTEM + _FEWSHOT_LAUDO_NOTE,
            messages=messages,
        )
        return response.content[0].text.strip() or laudo_bruto
    except Exception:
        return polish_laudo(laudo_bruto, resumo_clinico)


def _laudo_examples(findings: ECGFindings) -> list[dict]:
    """Recupera exemplos para o fewshot do laudo. Nunca derruba o pipeline."""
    try:
        from synthesis import fewshot
        try:
            padroes = classify(findings)
        except Exception:
            padroes = []
        try:
            risco, _, _ = compute_risk(padroes, PatientData())
        except Exception:
            risco = ""
        return fewshot.retrieve_laudo(padroes, risco, FEWSHOT_K)
    except Exception:
        return []


def generate_laudo(
    findings: ECGFindings,
    paciente: PatientData | None = None,
    polish: bool = True,
) -> str:
    """Gera laudo final. `paciente` mantido por compat — polish não recebe."""
    bruto = laudo_from_template(findings)
    if not polish:
        return bruto
    if FEWSHOT_LAUDO:
        exemplos = _laudo_examples(findings)
        if exemplos:
            return polish_laudo_with_examples(bruto, findings.resumo_clinico, exemplos)
    return polish_laudo(bruto, findings.resumo_clinico)
