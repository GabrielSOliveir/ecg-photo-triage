import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import PatientData, SinaisVitais
from rules.risk_engine import compute_risk, base_risk_from_padroes


def _p(queixa="", **sv) -> PatientData:
    return PatientData(queixa=queixa, sinais_vitais=SinaisVitais(**sv))


def test_base_risk_max():
    assert base_risk_from_padroes(["ecg_normal"]) == "BAIXO"
    assert base_risk_from_padroes(["stemi_anterior"]) == "EMERGENCIA"
    assert base_risk_from_padroes(["fa_alta_resposta", "bre"]) == "INTERMEDIARIO"


def test_normal_assintomatico_baixo():
    r, _, _ = compute_risk(["ecg_normal"], _p(queixa="check-up"))
    assert r == "BAIXO"


def test_normal_com_dor_intermediario():
    r, _, regras = compute_risk(["ecg_normal"], _p(queixa="dor torácica há 2h"))
    assert r == "INTERMEDIARIO"
    assert any("queixa_critica" in x for x in regras)


def test_stemi_sempre_emergencia():
    r, _, _ = compute_risk(["stemi_anterior"], _p(queixa="check-up"))
    assert r == "EMERGENCIA"


def test_vitais_sobrepoem():
    # ECG normal mas paciente em choque
    r, alertas, _ = compute_risk(["ecg_normal"], _p(pa_sistolica=70))
    assert r == "EMERGENCIA"
    assert len(alertas) == 1


def test_vitais_e_padrao_pegam_max():
    # FA com alta resposta + hipotensão
    r, alertas, _ = compute_risk(["fa_alta_resposta"], _p(pa_sistolica=80))
    assert r == "ALTO"


def test_bav_3g_emergencia():
    r, _, _ = compute_risk(["bav_3g"], _p())
    assert r == "EMERGENCIA"


def test_alertas_propagados():
    _, alertas, regras = compute_risk(["ecg_normal"], _p(pa_sistolica=80, sat=85))
    assert len(alertas) == 2
    assert "piso_vitais=ALTO" in " ".join(regras)


def test_sem_padroes_e_sem_sintomas():
    r, _, _ = compute_risk([], _p())
    assert r == "BAIXO"
