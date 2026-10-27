"""Orquestrador do pipeline ECG.

Etapas (MVP):
  0. Validação da entrada
  1. Vision extraction (Sonnet + tool use)
  2. Quality gate
  3. Pattern classification (Python)
  4. Risk engine (Python)
  5. Laudo composition (template + Haiku polish)
  6. Conduta (Sonnet zero-shot)
  7. Response assembly"""

import sys
import time
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PIPELINE_VERSION
from schemas import (
    PatientData,
    ECGResponse,
    Justificativa,
    ECGFindings,
)
from vision.extractor import extract_findings, VisionExtractionError
from rules.pattern_classifier import classify
from rules.risk_engine import compute_risk
from synthesis.laudo import generate_laudo
from synthesis.conduta import generate_conduta


def analyze(
    image_source: Union[str, Path, bytes],
    patient: PatientData,
    polish_laudo: bool = True,
    vision_model: str | None = None,
    conduta_model: str | None = None,
) -> ECGResponse:
    t0 = time.time()

    # ── 1. Vision extraction ──────────────────────────────────────────────
    try:
        findings: ECGFindings = extract_findings(image_source, patient, model=vision_model)
    except VisionExtractionError as e:
        return ECGResponse(
            status="erro",
            erro=str(e),
            pipeline_version=PIPELINE_VERSION,
            tempo_ms=int((time.time() - t0) * 1000),
        )

    # ── 2. Quality gate ────────────────────────────────────────────────────
    if findings.qualidade.score_global == "INVALIDA":
        return ECGResponse(
            status="inconclusivo",
            erro=findings.qualidade.observacoes or "Imagem inadequada para interpretação",
            confianca_imagem=findings.qualidade.confiabilidade,
            confianca_global=0.0,
            findings=findings,
            pipeline_version=PIPELINE_VERSION,
            tempo_ms=int((time.time() - t0) * 1000),
        )

    # ── 3. Pattern classification ─────────────────────────────────────────
    padroes = classify(findings)

    # ── 4. Risk engine ────────────────────────────────────────────────────
    risco, alertas, regras = compute_risk(padroes, patient)
    alertas_textos = [a.motivo for a in alertas]

    # ── 5. Laudo ──────────────────────────────────────────────────────────
    laudo = generate_laudo(findings, patient, polish=polish_laudo)

    # ── 6. Conduta ────────────────────────────────────────────────────────
    conduta = generate_conduta(laudo, findings, patient, risco, alertas_textos, model=conduta_model)

    # ── 7. Confiança global ───────────────────────────────────────────────
    conf_img = findings.qualidade.confiabilidade
    # Confiança global: imagem × penalidade por qualidade degradada
    penalidade = {
        "BOA": 1.0,
        "PARCIAL": 0.85,
        "DEGRADADA": 0.7,
        "INVALIDA": 0.0,
    }.get(findings.qualidade.score_global, 0.5)
    conf_global = round(conf_img * penalidade, 2)

    achados_chave: list[str] = []
    if padroes != ["ecg_normal"]:
        achados_chave = padroes[:5]

    return ECGResponse(
        status="ok",
        laudo=laudo,
        conduta=conduta,
        risco=risco,
        confianca_imagem=conf_img,
        confianca_global=conf_global,
        findings=findings,
        justificativa=Justificativa(
            padroes_identificados=padroes,
            regras_disparadas=regras,
            achados_chave=achados_chave,
            alertas_vitais=alertas_textos,
        ),
        pipeline_version=PIPELINE_VERSION,
        tempo_ms=int((time.time() - t0) * 1000),
    )
