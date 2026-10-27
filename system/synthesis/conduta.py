"""Conduta — Sonnet zero-shot tendo findings + clínica + risco computado como input.

Decisão: não usar RAG no MVP. O baseline do README mostra que zero-shot Sonnet supera RAG no estilo das médicas. Quando o feedback acumular casos, dá pra ir refinando via fewshot dinâmico."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from config import ANTHROPIC_API_KEY, CONDUTA_MODEL, FEWSHOT_CONDUTA, FEWSHOT_K
from schemas import ECGFindings, PatientData, Risco
from synthesis import fewshot
from rules.pattern_classifier import classify

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """Você é plantonista de PS brasileiro. CONSERVADOR. Sua referência de estilo é
o que uma médica generalista escreveria em prontuário de PS — não o que um cardiologista
escreveria num atendimento eletivo.

Princípio: **NÃO ANTECIPE diagnóstico**. Confirme com exame complementar antes de escalar.
- Dor torácica + ECG suspeito → "troponina seriada 0h/3h, repetir ECG, observar"
- NÃO pular para "AAS 300mg + clopidogrel + heparina + acionar hemodinâmica" sem troponina elevada e ECG claramente diagnóstico.

REGRAS:
- Estilo prontuário direto, telegráfico. Sem didatismo, sem explicações.
- Máximo 2 linhas (~200 caracteres total).
- Conduta concreta: exame + janela, medicação sintomática, observação, alta/encaminhamento.
- Não repita o laudo. Não explique o diagnóstico.

CALIBRAÇÃO POR RISCO (use como teto, não como obrigação):
- EMERGENCIA → só se ECG claramente diagnóstico (STEMI inequívoco em 2+ derivações contíguas,
  TV sustentada, BAV total, FV) + clínica compatível. "Acionar hemodinâmica/UTI/protocolo X".
- ALTO → exames imediatos + observação contínua. Considerar contato cardio.
- INTERMEDIARIO (padrão para dor torácica com ECG não-conclusivo):
  "Troponina seriada 0h/3h. Repetir ECG em 15-30min. Observar XYh."
- BAIXO → orientações, sintomáticos, alta com retorno.

QUANDO QUALIDADE DO ECG FOR PARCIAL/DEGRADADA:
- Não acione protocolos baseados em achados duvidosos.
- Conduta priorita REPETIR ECG com técnica adequada + observar.

Se houver alertas vitais críticos (PA<90, Sat<90, FC<40), aborde-os explicitamente.

Português. Sem hedges desnecessários ("considerar", "avaliar" só onde apropriado)."""


USER_TEMPLATE = """LAUDO ECG (estruturado):
{laudo}

DADOS DO PACIENTE:
{paciente}

QUALIDADE DA IMAGEM: {qualidade}  (confiabilidade {conf:.0%})
RISCO CALCULADO: {risco}
ALERTAS VITAIS: {alertas}

Gere apenas a conduta. Sem cabeçalhos."""


# Nota anexada ao system quando há exemplos few-shot. Reforça: copie o ESTILO,
# não o CONTEÚDO — a conduta deve sair do laudo/caso atual.
_FEWSHOT_NOTE = (
    "\n\nVocê verá abaixo exemplos de casos anteriores com a conduta no estilo "
    "desejado (voz da médica). IMITE O ESTILO desses exemplos — estrutura, "
    "concisão, expressões recorrentes. O CONTEÚDO da conduta deve sair do LAUDO "
    "e do CASO ATUAL; não copie o conteúdo clínico dos exemplos."
)

# Template enxuto dos exemplos (o corpus só tem caso+laudo+conduta, sem risco/qualidade).
_EXEMPLO_USER = """LAUDO ECG:
{laudo}

CASO: {caso}

Gere apenas a conduta. Sem cabeçalhos."""


def _montar_fewshot(patient: PatientData, laudo_texto: str,
                    findings: ECGFindings, risco: Risco) -> list[dict]:
    """Recupera exemplos pelo QUADRO (padrões + risco + queixa) e os formata como
    turnos user/assistant. [] se nada casar ou se a recuperação falhar (o pipeline
    nunca quebra por causa do few-shot)."""
    try:
        query = " ".join(p for p in (patient.queixa, patient.contexto) if p).strip()
        try:
            padroes = classify(findings)
        except Exception:
            padroes = []
        exemplos = fewshot.retrieve(query or laudo_texto, padroes, str(risco), FEWSHOT_K)
        msgs: list[dict] = []
        for ex in exemplos:
            msgs.append({"role": "user",
                         "content": _EXEMPLO_USER.format(laudo=ex["laudo"], caso=ex["caso"])})
            msgs.append({"role": "assistant", "content": ex["conduta"]})
        return msgs
    except Exception:
        return []


def generate_conduta(
    laudo_texto: str,
    findings: ECGFindings,
    patient: PatientData,
    risco: Risco,
    alertas_vitais: list[str],
    model: str | None = None,
) -> str:
    alertas_str = "; ".join(alertas_vitais) if alertas_vitais else "nenhum"
    user = USER_TEMPLATE.format(
        laudo=laudo_texto,
        paciente=patient.resumo(),
        qualidade=findings.qualidade.score_global,
        conf=findings.qualidade.confiabilidade,
        risco=risco,
        alertas=alertas_str,
    )
    system = SYSTEM_PROMPT
    messages: list[dict] = [{"role": "user", "content": user}]
    if FEWSHOT_CONDUTA:
        exemplos_msgs = _montar_fewshot(patient, laudo_texto, findings, risco)
        if exemplos_msgs:
            system = SYSTEM_PROMPT + _FEWSHOT_NOTE
            messages = exemplos_msgs + messages

    try:
        # Sonnet 5 / Opus 4.7+ rejeitam temperature (400); só enviamos em modelos
        # que aceitam (haiku, sonnet-4-6). O system prompt já é bem prescritivo.
        create_kwargs: dict = {
            "model": model or CONDUTA_MODEL,
            "max_tokens": 300,
            "system": system,
            "messages": messages,
        }
        m = create_kwargs["model"]
        if not (m.startswith("claude-sonnet-5") or m.startswith("claude-opus-4-7")
                or m.startswith("claude-opus-4-8") or m.startswith("claude-fable")):
            create_kwargs["temperature"] = 0
        response = _get_client().messages.create(**create_kwargs)
        return response.content[0].text.strip()
    except Exception as e:
        return f"(Falha ao gerar conduta: {e})"
