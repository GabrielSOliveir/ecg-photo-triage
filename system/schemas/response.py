from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

from schemas.findings import ECGFindings

Risco = Literal["BAIXO", "INTERMEDIARIO", "ALTO", "EMERGENCIA", "INDETERMINADO"]

StatusResposta = Literal["ok", "inconclusivo", "erro"]

ComparisonCategoria = Literal[
    "AGREE",
    "CATCH",
    "MISS",
    "HALLUCINATION",
    "NEUTRAL_DISAGREE",
]


class Justificativa(BaseModel):
    model_config = ConfigDict(extra="forbid")
    padroes_identificados: list[str] = Field(default_factory=list)
    regras_disparadas: list[str] = Field(default_factory=list)
    achados_chave: list[str] = Field(default_factory=list)
    alertas_vitais: list[str] = Field(default_factory=list)


class ECGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StatusResposta = "ok"
    erro: Optional[str] = None

    laudo: str = ""
    conduta: str = ""
    risco: Risco = "INDETERMINADO"

    confianca_imagem: float = Field(0.0, ge=0.0, le=1.0)
    confianca_global: float = Field(0.0, ge=0.0, le=1.0)

    findings: Optional[ECGFindings] = None
    justificativa: Justificativa = Field(default_factory=Justificativa)

    pipeline_version: str = ""
    tempo_ms: int = 0

    def whatsapp(self) -> str:
        if self.status == "inconclusivo":
            return f"*INCONCLUSIVO* — {self.erro or 'imagem inadequada'}"
        if self.status == "erro":
            return f"⚠️ *ERRO* — {self.erro or 'falha no processamento'}"

        lines = [
            "*LAUDO*",
            self.laudo,
            "",
            "*CONDUTA*",
            self.conduta,
        ]
        if self.risco in ("ALTO", "EMERGENCIA"):
            alertas = self.justificativa.alertas_vitais
            if alertas:
                lines.append(f"⚠️ {'; '.join(alertas)}")
        risco_label = "INTERMEDIÁRIO" if self.risco == "INTERMEDIARIO" else (
            "EMERGÊNCIA" if self.risco == "EMERGENCIA" else self.risco
        )
        lines.append(f"*RISCO:* {risco_label}")
        lines.append(f"*Conf:* {int(self.confianca_global * 100)}%")
        return "\n".join(lines)


class ComparisonItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campo: str
    categoria: ComparisonCategoria
    sistema: str = ""
    gt: str = ""
    peso_clinico: Literal["baixo", "medio", "alto"] = "medio"


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: int
    itens: list[ComparisonItem] = Field(default_factory=list)

    @property
    def n_agree(self) -> int:
        return sum(1 for i in self.itens if i.categoria == "AGREE")

    @property
    def n_catch(self) -> int:
        return sum(1 for i in self.itens if i.categoria == "CATCH")

    @property
    def n_miss(self) -> int:
        return sum(1 for i in self.itens if i.categoria == "MISS")

    @property
    def n_hallucination(self) -> int:
        return sum(1 for i in self.itens if i.categoria == "HALLUCINATION")

    @property
    def n_neutral_disagree(self) -> int:
        return sum(1 for i in self.itens if i.categoria == "NEUTRAL_DISAGREE")

    def has_clinical_disagreement(self) -> bool:
        return any(
            i.categoria in ("CATCH", "MISS", "HALLUCINATION")
            and i.peso_clinico in ("medio", "alto")
            for i in self.itens
        )
