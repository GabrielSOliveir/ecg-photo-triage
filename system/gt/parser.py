"""Parser de ground truth: texto livre do laudo da médica → ECGFindings estruturado.

Estratégia: tool use com Haiku (rápido + barato). Resultado cacheado por case_id.

Limitação: textos curtos como "ECG dentro da normalidade" produzem findings com tudo default.
Isso é OK — o comparator trata ausência == ausência."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from config import ANTHROPIC_API_KEY, GT_PARSE_MODEL
from schemas import ECGFindings

GT_PARSER_VERSION = "gt-parser-v1"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """Você extrai achados estruturados de laudos de ECG escritos em português por médicos brasileiros.

O texto é curto, telegráfico, estilo prontuário. Exemplos típicos:
- "Taquicardia sinusal, sem sinais de isquemia aguda"
- "Sinusal Eixo preservado Sem sobrecargas Onda T mais apiculada"
- "MP com condução ventricular sem aparentes falhas de comando"
- "ECG dentro da normalidade"

REGRAS:
- Preencha APENAS o que está explícito no texto. Não infira.
- Se texto diz "sinusal" → ritmo=sinusal; sem isso → indeterminado.
- Se texto não menciona FC → fc_bpm=0.
- Se diz "sem sobrecargas" / "sem alterações" → mantenha booleanos como False (default).
- "MP" / "marcapasso" → ritmo=marcapasso, marcapasso=true.
- Para o campo qualidade, sempre use score_global="BOA" e confiabilidade=1.0 (GT é referência, não tem qualidade de imagem a julgar).
- resumo_clinico: cole o texto original do laudo aqui.

Use o tool extract_ecg_findings."""


class GTParseError(Exception):
    pass


def parse_gt_laudo(laudo_texto: str) -> ECGFindings:
    if not laudo_texto or not laudo_texto.strip():
        return ECGFindings(resumo_clinico="")

    schema = ECGFindings.model_json_schema()
    tool = {
        "name": "extract_ecg_findings",
        "description": "Estrutura achados de ECG.",
        "input_schema": schema,
    }

    try:
        response = _get_client().messages.create(
            model=GT_PARSE_MODEL,
            max_tokens=2048,
            temperature=0,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "extract_ecg_findings"},
            messages=[{
                "role": "user",
                "content": f"Laudo:\n{laudo_texto}\n\nExtraia via tool.",
            }],
        )
    except anthropic.APIError as e:
        raise GTParseError(f"API Anthropic falhou: {e}") from e

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_ecg_findings":
            try:
                return ECGFindings.model_validate(block.input)
            except Exception as e:
                raise GTParseError(f"Tool retornou JSON inválido: {e}") from e

    raise GTParseError(f"Modelo não invocou a tool. stop_reason={response.stop_reason}")
