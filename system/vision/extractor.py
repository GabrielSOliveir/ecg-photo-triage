"""Vision extractor — Sonnet com tool use força ECGFindings garantido.

Sem fallback de schema vazio. Se a extração falhar, levanta exceção explícita."""

import base64
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from config import ANTHROPIC_API_KEY, VISION_MODEL, VISION_THINKING, VISION_EFFORT
from schemas import ECGFindings, PatientData
from vision.preprocess import preprocess

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY não configurada (.env)")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _encode(image_source: Union[str, Path, bytes]) -> tuple[str, str]:
    """Lê os bytes da imagem e aplica o preprocess clássico antes de devolver
    base64. Preprocess sempre devolve JPEG (mesmo se a entrada era PNG/WebP),
    então o media_type abaixo é sempre image/jpeg. Se PREPROCESS_VISION=0,
    devolve a imagem original (mas ainda como JPEG via re-encode opcional)."""
    if isinstance(image_source, bytes):
        raw = image_source
    else:
        raw = Path(image_source).read_bytes()
    processed = preprocess(raw)
    return base64.standard_b64encode(processed).decode(), "image/jpeg"


SYSTEM_PROMPT = """Você é cardiologista experiente em eletrocardiografia. Analise a foto e os dados clínicos. Preencha o tool `extract_ecg_findings`.

POLÍTICA POR TIPO DE ACHADO — bar diferente para cada categoria:

═══ BAR ALTO (não reporte se em dúvida — falso positivo cascateia em IAM/protocolo) ═══

1. ST SUPRA — REPORTE APENAS SE TODOS:
   (a) magnitude ≥2mm em V1-V3 ou ≥1mm nas demais
   (b) em ≥2 derivações contíguas (DII+DIII+aVF / V1-V3 / V4-V6 / DI+aVL+V5-V6)
   (c) morfologia convexa ou horizontal (NÃO côncava — pode ser repolarização precoce)
   (d) a derivação NÃO está em derivacoes_problematicas
   Falhou qualquer? derivacoes=[], magnitude=0.

2. ST INFRA — REPORTE APENAS SE TODOS:
   (a) magnitude ≥1mm, em ≥2 derivações contíguas
   (b) horizontal ou descendente (ascendente lento em taquicardia NÃO conta)
   (c) qualidade BOA da derivação

3. BLOQUEIOS — critério morfológico clássico (BRE: QRS≥120ms+R alargado em V5-V6+ausência Q DI; BRD: rsR' em V1+S alargado em DI/V6; etc.) ou NÃO reporte.

4. ACHADOS ESPECIAIS de protocolo (Sgarbossa, Wellens, De Winter, Brugada, S1Q3T3) — só com critério clássico evidente.

═══ BAR MODERADO (REPORTE quando observar, mesmo sutil — é descritivo) ═══

5. ONDA T (campo onda_t_alteracoes): liste as derivações onde a onda T está invertida/achatada/apiculada/bifásica/profunda simétrica, mesmo se a alteração for discreta. É descritivo, não diagnóstico definitivo. Se T está claramente diferente do esperado em ≥1 derivação, REPORTE.

6. SOBRECARGAS — quando há sinais sugestivos, marque true:
   - hipertrofia_ve: R em aVL >11mm, índice Sokolow-Lyon ≥35mm, padrão de strain lateral. Mesmo se borderline, marque.
   - hipertrofia_vd: R>S em V1, desvio direito + R alta em V1.
   - sobrecarga_atrial_esq: P mitrale (>120ms ou bífida em DII, P negativa profunda em V1).
   - sobrecarga_atrial_dir: P pulmonale (apiculada >2,5mm em DII/DIII/aVF).

7. ÁREAS INATIVAS: quando há QS ou perda de R/baixa amplitude em parede esperada, liste a parede. No resumo_clinico, qualifique com "sugere área inativa em X" se duvidoso por posicionamento de eletrodo.

8. ALTERAÇÃO INESPECÍFICA DE REPOLARIZAÇÃO: quando há alterações de ST/T em múltiplas derivações sem padrão isquêmico definido E sem padrão específico identificável, use achados_especiais=["alt_repolarizacao_difusa"].
   IMPORTANTE — PRIORIDADE DOS PADRÕES ESPECÍFICOS: se o contexto clínico + traçado sugerirem um padrão NOMEADO, PREFIRA o padrão específico em vez de "alt_repolarizacao_difusa". Exemplos:
     - T achatadas/U proeminentes + K+ baixo na clínica → `hipocalemia_padrao`
     - T apiculadas simétricas + K+ alto / DRC → `hipercalemia_padrao`
     - S1Q3T3 + suspeita de TEP → `s1q3t3`
     - Strain lateral + HVE evidente → marque `hipertrofia_ve=true` (não use só "alt_repolarizacao_difusa")
   `alt_repolarizacao_difusa` é a categoria de ÚLTIMO recurso quando NENHUM padrão específico se aplica. Não a use pra "engolir" padrões nomeados.

9. EIXO: reporte desvio mesmo se sutil.

═══ OBRIGATÓRIOS QUANDO POSSÍVEL ═══

10. RITMO: sinusal só se P sinusal antecedendo cada QRS com PR constante. Senão escolha o que melhor descreve (FA, juncional, etc.) ou "indeterminado".

11. FC: estime pelo R-R. Se irregular, FC média + fc_variavel=true.

12. INTERVALOS: PR/QRS/QTc (Bazett) quando mensuráveis. 0 só se realmente incerto.

═══ RESUMO CLÍNICO (campo resumo_clinico) ═══

2-4 frases descritivas no estilo da médica que assina os ECGs deste serviço.
Esse texto vira matéria-prima do laudo final — descreva tudo que enxergou em
linguagem de prontuário, mesmo que o schema acima já tenha capturado parte.

QUANDO APLICÁVEL, inclua estes descritores qualitativos (eles são parte do
estilo dela e o schema estruturado não captura):

(a) VOLTAGEM: "baixa voltagem nas derivações frontais/precordiais",
    "voltagens preservadas".
(b) STRAIN: quando há HVE/HVD, descreva "padrão de strain lateral" ou
    "padrão de strain inferior" — i.e., infra de ST com T invertida na parede
    da câmara sobrecarregada.
(c) EVOLUÇÃO TEMPORAL APARENTE: em supra de ST, comente "aspecto de poucas
    horas / pelo menos X horas / dias de evolução" se a morfologia (Q
    patológica, onda T) sugerir. Use o qualificador "aparente"/"compatível
    com" — não dá pra cravar tempo sem ECG prévio.
(d) SECUNDÁRIO A: para alterações de ST/T em taquicardia/bradicardia/HVE,
    sinalize "(secundárias a taquicardia)", "(secundárias a sobrecarga)" —
    distingue alteração inespecífica de isquemia.
(e) SIGLAS EXPANDIDAS: a primeira ocorrência de cada sigla clínica vem com a
    expansão entre parênteses. Exemplos: "HVE (hipertrofia ventricular
    esquerda)", "BRE (bloqueio de ramo esquerdo)", "FA (fibrilação atrial)".

TAMBÉM PODE:
- usar qualificadores: "sugere", "compatível com", "discreta", "padrão de",
  "alteração inespecífica"
- cruzar com dados do paciente quando relevante ("em contexto de dor torácica
  + troponina elevada, considerar isquemia subendocárdica")
- reconhecer incerteza: "alteração possivelmente por posicionamento de eletrodo"

NÃO inclua conduta — descrição/interpretação apenas. NÃO invente achado que
você não enxergou no traçado só pra usar um descritor da lista.

═══ QUALIDADE ═══

Avalie por derivação. Dobra/curvatura/borrão → PARCIAL ou DEGRADADA. INVALIDA só se ilegível. Liste derivacoes_problematicas; nenhum achado de ST supra/infra deve depender de derivação problemática.

LEMBRE: a médica quer ver achados clínicos descritos como em planilha de prontuário (alteração de repolarização, sobrecarga, área inativa, T invertida) — não só métricas (FC/PR/QRS/QTc). Soft findings com qualificador são esperados quando observados."""


USER_TEMPLATE = """Foto do ECG anexada. Dados clínicos:

{paciente_resumo}

Extraia os achados estruturados via tool."""


class VisionExtractionError(Exception):
    pass


def extract_findings(
    image_source: Union[str, Path, bytes],
    patient: PatientData,
    model: str | None = None,
) -> ECGFindings:
    """Chama o modelo com tool use. Retorna ECGFindings validado ou levanta VisionExtractionError."""
    b64, media_type = _encode(image_source)
    schema = ECGFindings.model_json_schema()

    tool = {
        "name": "extract_ecg_findings",
        "description": "Estrutura os achados do ECG fotografado em campos padronizados.",
        "input_schema": schema,
    }

    # Com thinking ligado a API NÃO aceita tool_choice forçado nem temperature.
    # Usamos tool_choice=auto (o SYSTEM_PROMPT já ordena chamar a tool) + effort.
    params: dict = {
        "model": model or VISION_MODEL,
        "max_tokens": 8192,
        "system": SYSTEM_PROMPT,
        "tools": [tool],
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {
                    "type": "text",
                    "text": USER_TEMPLATE.format(paciente_resumo=patient.resumo()),
                },
            ],
        }],
    }
    if VISION_THINKING:
        params["thinking"] = {"type": "adaptive"}
        params["extra_body"] = {"output_config": {"effort": VISION_EFFORT}}
        # tool_choice auto (default) — forçar tool é incompatível com thinking
    else:
        params["temperature"] = 0
        params["tool_choice"] = {"type": "tool", "name": "extract_ecg_findings"}

    try:
        response = _get_client().messages.create(**params)
    except anthropic.APIError as e:
        raise VisionExtractionError(f"API Anthropic falhou: {e}") from e

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_ecg_findings":
            inp = block.input
            # o modelo às vezes aninha tudo sob 1 chave (ecg_findings/findings) — desembrulha
            if isinstance(inp, dict) and len(inp) == 1 and isinstance(next(iter(inp.values())), dict):
                inp = next(iter(inp.values()))
            try:
                return ECGFindings.model_validate(inp)
            except Exception as e:
                raise VisionExtractionError(
                    f"Tool retornou JSON inválido para ECGFindings: {e}"
                ) from e

    raise VisionExtractionError(
        f"Modelo não invocou a tool. stop_reason={response.stop_reason}"
    )
