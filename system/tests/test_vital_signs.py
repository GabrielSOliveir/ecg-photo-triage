import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import PatientData, SinaisVitais
from rules.vital_signs import vital_floor, queixa_critica, max_risco


def _p(**sv) -> PatientData:
    return PatientData(sinais_vitais=SinaisVitais(**sv))


def test_no_vitals_returns_indeterminado():
    risco, alertas = vital_floor(PatientData())
    assert risco == "INDETERMINADO"
    assert alertas == []


def test_pa_choque_emergencia():
    risco, alertas = vital_floor(_p(pa_sistolica=65, pa_diastolica=40))
    assert risco == "EMERGENCIA"
    assert any("choque" in a.motivo.lower() for a in alertas)


def test_pa_hipotensao_alto():
    risco, _ = vital_floor(_p(pa_sistolica=85))
    assert risco == "ALTO"


def test_pa_emergencia_hipertensiva():
    risco, _ = vital_floor(_p(pa_sistolica=230))
    assert risco == "ALTO"


def test_pa_hipertensao_grave_intermediario():
    risco, _ = vital_floor(_p(pa_sistolica=185))
    assert risco == "INTERMEDIARIO"


def test_pa_normal_indeterminado():
    risco, _ = vital_floor(_p(pa_sistolica=130, pa_diastolica=80))
    assert risco == "INDETERMINADO"


def test_fc_extremo_emergencia():
    assert vital_floor(_p(fc=30))[0] == "EMERGENCIA"
    assert vital_floor(_p(fc=200))[0] == "EMERGENCIA"


def test_fc_alto():
    assert vital_floor(_p(fc=42))[0] == "ALTO"
    assert vital_floor(_p(fc=160))[0] == "ALTO"


def test_sat_grave():
    assert vital_floor(_p(sat=82))[0] == "EMERGENCIA"
    assert vital_floor(_p(sat=88))[0] == "ALTO"
    assert vital_floor(_p(sat=92))[0] == "INTERMEDIARIO"


def test_fr_alto():
    assert vital_floor(_p(fr=32))[0] == "ALTO"
    assert vital_floor(_p(fr=6))[0] == "ALTO"


def test_combo_pega_maior():
    risco, alertas = vital_floor(_p(pa_sistolica=180, sat=83))
    assert risco == "EMERGENCIA"
    assert len(alertas) == 2


def test_queixa_critica():
    assert queixa_critica("dor torácica em aperto")
    assert queixa_critica("dispneia aos esforços")
    assert queixa_critica("síncope na rua")
    assert not queixa_critica("dor lombar")
    assert not queixa_critica("")
    assert not queixa_critica("check-up de rotina")


def test_max_risco():
    assert max_risco(["BAIXO", "ALTO", "INTERMEDIARIO"]) == "ALTO"
    assert max_risco(["EMERGENCIA", "BAIXO"]) == "EMERGENCIA"
    assert max_risco([]) == "INDETERMINADO"
