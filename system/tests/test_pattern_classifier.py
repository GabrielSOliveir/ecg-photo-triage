import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import ECGFindings, STDesvio, OndaTAlteracao
from rules.pattern_classifier import classify


def test_normal():
    f = ECGFindings(ritmo="sinusal", fc_bpm=72, eixo="normal")
    assert classify(f) == ["ecg_normal"]


def test_taquicardia_sinusal():
    f = ECGFindings(ritmo="sinusal", fc_bpm=120)
    pads = classify(f)
    assert "taquicardia_sinusal" in pads


def test_bradicardia_sinusal():
    f = ECGFindings(ritmo="sinusal", fc_bpm=42)
    pads = classify(f)
    assert "bradicardia_sinusal" in pads


def test_fa_alta_resposta():
    f = ECGFindings(ritmo="fa", fc_bpm=145)
    pads = classify(f)
    assert "fa_alta_resposta" in pads
    assert "fibrilacao_atrial" not in pads


def test_fa_normocard():
    f = ECGFindings(ritmo="fa", fc_bpm=85)
    pads = classify(f)
    assert "fibrilacao_atrial" in pads


def test_stemi_anterior():
    f = ECGFindings(
        ritmo="sinusal", fc_bpm=80,
        st_supra=STDesvio(derivacoes=["V2", "V3", "V4"], magnitude_mm_max=3.0),
    )
    pads = classify(f)
    assert "stemi_anterior" in pads


def test_stemi_inferior():
    f = ECGFindings(
        ritmo="sinusal", fc_bpm=80,
        st_supra=STDesvio(derivacoes=["DII", "DIII", "aVF"], magnitude_mm_max=2.0),
    )
    pads = classify(f)
    assert "stemi_inferior" in pads


def test_infra_st():
    f = ECGFindings(
        ritmo="sinusal", fc_bpm=80,
        st_infra=STDesvio(derivacoes=["V4", "V5", "V6"], magnitude_mm_max=1.5),
    )
    pads = classify(f)
    assert "infra_st_isquemico" in pads


def test_bav_3g():
    f = ECGFindings(ritmo="sinusal", fc_bpm=40, bloqueios=["bav_3g"])
    pads = classify(f)
    assert "bav_3g" in pads
    assert "bav_total" in pads


def test_bre():
    f = ECGFindings(ritmo="sinusal", fc_bpm=72, bloqueios=["bre"])
    pads = classify(f)
    assert "bre" in pads
    assert "bloqueio_ramo_esquerdo" in pads


def test_qt_longo():
    f = ECGFindings(ritmo="sinusal", fc_bpm=72, qtc_ms=510)
    assert "qt_muito_longo" in classify(f)


def test_qt_intermediario():
    f = ECGFindings(ritmo="sinusal", fc_bpm=72, qtc_ms=470)
    assert "qt_longo" in classify(f)


def test_inversao_t():
    f = ECGFindings(
        ritmo="sinusal", fc_bpm=72,
        onda_t_alteracoes=[OndaTAlteracao(tipo="invertida", derivacoes=["V1", "V2", "V3"])],
    )
    assert "inversao_onda_t" in classify(f)


def test_wpw():
    f = ECGFindings(ritmo="sinusal", fc_bpm=72, pre_excitacao=True)
    assert "pre_excitacao_wpw" in classify(f)


def test_marcapasso():
    f = ECGFindings(ritmo="marcapasso", fc_bpm=70, marcapasso=True)
    pads = classify(f)
    assert "ritmo_marcapasso" in pads
    assert pads.count("ritmo_marcapasso") == 1


def test_multiplos_padroes():
    f = ECGFindings(
        ritmo="fa", fc_bpm=130, bloqueios=["bre"],
    )
    pads = classify(f)
    assert "fa_alta_resposta" in pads
    assert "bre" in pads
