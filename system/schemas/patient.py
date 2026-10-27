from typing import Optional, Literal
from pydantic import BaseModel, Field


class SinaisVitais(BaseModel):
    pa_sistolica: Optional[int] = Field(None, description="mmHg")
    pa_diastolica: Optional[int] = Field(None, description="mmHg")
    fc: Optional[int] = Field(None, description="bpm")
    sat: Optional[int] = Field(None, ge=0, le=100, description="SpO2 %")
    fr: Optional[int] = Field(None, description="irpm")
    temperatura: Optional[float] = Field(None, description="°C")

    @classmethod
    def from_pa_string(cls, pa: str) -> tuple[Optional[int], Optional[int]]:
        if not pa or "/" not in pa:
            return None, None
        try:
            s, d = pa.split("/", 1)
            return int(s.strip()), int(d.strip())
        except (ValueError, AttributeError):
            return None, None


class PatientData(BaseModel):
    idade: Optional[int] = None
    sexo: Optional[Literal["M", "F"]] = None
    queixa: str = ""
    comorbidades: list[str] = []
    sinais_vitais: SinaisVitais = SinaisVitais()
    contexto: str = ""

    def resumo(self) -> str:
        partes = []
        if self.idade is not None and self.sexo:
            sexo_str = "masc" if self.sexo == "M" else "fem"
            partes.append(f"{self.idade}a {sexo_str}")
        elif self.idade is not None:
            partes.append(f"{self.idade}a")
        if self.comorbidades:
            partes.append(", ".join(self.comorbidades))
        if self.queixa:
            partes.append(f"queixa: {self.queixa}")
        sv = self.sinais_vitais
        if sv.pa_sistolica:
            pa = f"{sv.pa_sistolica}/{sv.pa_diastolica or '?'}"
            partes.append(f"PA {pa}")
        if sv.fc:
            partes.append(f"FC {sv.fc}")
        if sv.sat:
            partes.append(f"Sat {sv.sat}%")
        return " | ".join(partes) if partes else "(dados clínicos não informados)"
