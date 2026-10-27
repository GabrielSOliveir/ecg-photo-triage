from schemas.patient import PatientData, SinaisVitais
from schemas.findings import (
    ECGFindings,
    QualidadeImagem,
    STDesvio,
    OndaTAlteracao,
    Derivacao,
    Parede,
    Ritmo,
    Bloqueio,
    AchadoEspecial,
    Eixo,
    QualidadeScore,
)
from schemas.response import (
    ECGResponse, Risco, Justificativa,
    ComparisonResult, ComparisonCategoria, ComparisonItem,
)

__all__ = [
    "PatientData",
    "SinaisVitais",
    "ECGFindings",
    "QualidadeImagem",
    "STDesvio",
    "OndaTAlteracao",
    "Derivacao",
    "Parede",
    "Ritmo",
    "Bloqueio",
    "AchadoEspecial",
    "Eixo",
    "QualidadeScore",
    "ECGResponse",
    "Risco",
    "Justificativa",
    "ComparisonResult",
    "ComparisonCategoria",
    "ComparisonItem",
]
