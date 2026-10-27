from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

Derivacao = Literal[
    "DI", "DII", "DIII", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

Parede = Literal[
    "anterior", "anterosseptal", "anterolateral",
    "inferior", "lateral_alta", "lateral_baixa", "posterior", "apical",
]

Ritmo = Literal[
    "sinusal", "fa", "flutter_atrial", "tv", "tsv", "ritmo_juncional",
    "ritmo_atrial_ectopico", "ritmo_idioventricular", "marcapasso",
    "fibrilacao_ventricular", "assistolia", "indeterminado",
]

Bloqueio = Literal[
    "bav_1g", "bav_2g_mob1", "bav_2g_mob2", "bav_3g",
    "bre", "brd", "bdas", "bdpi",
]

AchadoEspecial = Literal[
    "sgarbossa_positivo", "wellens", "de_winter", "brugada",
    "s1q3t3", "qt_longo", "qt_curto", "tako_tsubo",
    "ritmo_juncional_acelerado", "pre_excitacao_wpw",
    "hipercalemia_padrao", "hipocalemia_padrao",
    "pericardite_padrao",
    "repolarizacao_precoce",
    "fibriloflutter",
    "alt_repolarizacao_difusa",
    "troca_eletrodos",
    "ritmo_atrial_multifocal",
]

Eixo = Literal[
    "normal", "desvio_esquerdo", "desvio_direito", "extremo", "indeterminado",
]

QualidadeScore = Literal["BOA", "PARCIAL", "DEGRADADA", "INVALIDA"]

MorfologiaST = Literal[
    "convexo_superior", "concavo_superior", "horizontal",
    "descendente", "ascendente_lento",
]

TipoOndaT = Literal[
    "invertida", "apiculada", "achatada", "bifasica_terminal_neg",
    "simetrica_profunda",
]


_DERIV_NORMALIZE = {
    "I": "DI", "II": "DII", "III": "DIII",
    "VR": "aVR", "VL": "aVL", "VF": "aVF",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
}


def _normalize_derivacoes(v):
    """Normaliza variantes brasileiras: 'III' → 'DIII', 'AVF' → 'aVF', etc."""
    if not isinstance(v, list):
        return v
    out = []
    for x in v:
        if isinstance(x, str):
            out.append(_DERIV_NORMALIZE.get(x.upper().strip(), x))
        else:
            out.append(x)
    return out


class STDesvio(BaseModel):
    model_config = ConfigDict(extra="forbid")
    derivacoes: list[Derivacao] = Field(default_factory=list)
    magnitude_mm_max: float = 0.0
    morfologia: Optional[MorfologiaST] = None

    @field_validator("derivacoes", mode="before")
    @classmethod
    def _norm(cls, v):
        return _normalize_derivacoes(v)


class OndaTAlteracao(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tipo: TipoOndaT
    derivacoes: list[Derivacao] = Field(default_factory=list)

    @field_validator("derivacoes", mode="before")
    @classmethod
    def _norm(cls, v):
        return _normalize_derivacoes(v)


class QualidadeImagem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score_global: QualidadeScore = "BOA"
    confiabilidade: float = Field(default=0.0, ge=0.0, le=1.0)
    derivacoes_problematicas: list[Derivacao] = Field(default_factory=list)
    observacoes: str = ""

    @field_validator("derivacoes_problematicas", mode="before")
    @classmethod
    def _norm(cls, v):
        return _normalize_derivacoes(v)


class ECGFindings(BaseModel):
    """Estrutura compartilhada — sistema produz E o GT parser produz no mesmo formato.
    O comparator opera campo a campo."""

    model_config = ConfigDict(extra="forbid")

    ritmo: Ritmo = "indeterminado"
    fc_bpm: int = 0
    fc_variavel: bool = False

    eixo: Eixo = "indeterminado"
    pr_ms: int = 0
    qrs_ms: int = 0
    qtc_ms: int = 0

    bloqueios: list[Bloqueio] = Field(default_factory=list)
    pre_excitacao: bool = False

    st_supra: STDesvio = Field(default_factory=STDesvio)
    st_infra: STDesvio = Field(default_factory=STDesvio)
    onda_t_alteracoes: list[OndaTAlteracao] = Field(default_factory=list)
    areas_inativas: list[Parede] = Field(default_factory=list)

    hipertrofia_ve: bool = False
    hipertrofia_vd: bool = False
    sobrecarga_atrial_esq: bool = False
    sobrecarga_atrial_dir: bool = False

    marcapasso: bool = False

    achados_especiais: list[AchadoEspecial] = Field(default_factory=list)

    qualidade: QualidadeImagem = Field(default_factory=QualidadeImagem)

    resumo_clinico: str = Field(
        default="",
        description="Resumo textual livre do interpretador (1-2 frases).",
    )

    def is_normal(self) -> bool:
        return (
            self.ritmo == "sinusal"
            and 50 <= self.fc_bpm <= 100
            and not self.bloqueios
            and not self.pre_excitacao
            and not self.st_supra.derivacoes
            and not self.st_infra.derivacoes
            and not self.onda_t_alteracoes
            and not self.areas_inativas
            and not self.hipertrofia_ve
            and not self.hipertrofia_vd
            and not self.marcapasso
            and not self.achados_especiais
            and (self.qtc_ms == 0 or self.qtc_ms <= 460)
        )
