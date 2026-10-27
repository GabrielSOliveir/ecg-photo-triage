import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import ECGFindings, STDesvio, OndaTAlteracao
from eval.comparator import compare


def _cat(result, campo, sistema_val=None) -> str | None:
    for it in result.itens:
        if it.campo == campo:
            if sistema_val is None or it.sistema == sistema_val:
                return it.categoria
    return None


def test_identical_normal():
    f = ECGFindings(ritmo="sinusal", fc_bpm=75, eixo="normal")
    r = compare(f, f, case_id=1)
    # Sinusal+75 → AGREE em ritmo + AGREE em fc
    assert r.n_agree == 2
    assert r.n_catch == 0
    assert r.n_miss == 0
    assert r.n_hallucination == 0


def test_system_catches_fa():
    sistema = ECGFindings(ritmo="fa", fc_bpm=130)
    gt = ECGFindings(ritmo="indeterminado", fc_bpm=0)
    r = compare(sistema, gt, case_id=1)
    assert _cat(r, "ritmo") == "CATCH"


def test_system_misses_stemi():
    sistema = ECGFindings(ritmo="sinusal", fc_bpm=80)
    gt = ECGFindings(
        ritmo="sinusal", fc_bpm=80,
        st_supra=STDesvio(derivacoes=["DII", "DIII", "aVF"], magnitude_mm_max=2.0),
    )
    r = compare(sistema, gt, case_id=1)
    miss_count = sum(1 for i in r.itens if i.categoria == "MISS" and i.campo == "st_supra")
    assert miss_count == 3  # cada derivação é um MISS


def test_hallucination_ritmo():
    sistema = ECGFindings(ritmo="tv", fc_bpm=180)
    gt = ECGFindings(ritmo="fa", fc_bpm=170)
    r = compare(sistema, gt, case_id=1)
    assert _cat(r, "ritmo") == "HALLUCINATION"


def test_fc_neutral_disagree():
    sistema = ECGFindings(ritmo="sinusal", fc_bpm=85)
    gt = ECGFindings(ritmo="sinusal", fc_bpm=78)
    r = compare(sistema, gt, case_id=1)
    fc_items = [i for i in r.itens if i.campo == "fc_bpm"]
    assert fc_items[0].categoria == "AGREE"


def test_fc_big_diff_hallucination():
    sistema = ECGFindings(ritmo="sinusal", fc_bpm=120)
    gt = ECGFindings(ritmo="sinusal", fc_bpm=70)
    r = compare(sistema, gt, case_id=1)
    fc_items = [i for i in r.itens if i.campo == "fc_bpm"]
    assert fc_items[0].categoria == "HALLUCINATION"


def test_bloqueios_overlap():
    sistema = ECGFindings(ritmo="sinusal", fc_bpm=80, bloqueios=["bre", "bav_1g"])
    gt = ECGFindings(ritmo="sinusal", fc_bpm=80, bloqueios=["bre"])
    r = compare(sistema, gt, case_id=1)
    cats = [(i.campo, i.categoria, i.sistema) for i in r.itens]
    assert ("bloqueios", "AGREE", "bre") in cats
    assert ("bloqueios", "CATCH", "bav_1g") in cats


def test_st_supra_partial_overlap():
    sistema = ECGFindings(
        ritmo="sinusal", fc_bpm=80,
        st_supra=STDesvio(derivacoes=["DII", "DIII", "aVF", "V6"], magnitude_mm_max=2.0),
    )
    gt = ECGFindings(
        ritmo="sinusal", fc_bpm=80,
        st_supra=STDesvio(derivacoes=["DII", "DIII", "aVF"], magnitude_mm_max=2.0),
    )
    r = compare(sistema, gt, case_id=1)
    catches = [i for i in r.itens if i.campo == "st_supra" and i.categoria == "CATCH"]
    assert len(catches) == 1
    assert catches[0].sistema == "V6"


def test_marcapasso_catch():
    sistema = ECGFindings(ritmo="marcapasso", fc_bpm=70, marcapasso=True)
    gt = ECGFindings(ritmo="indeterminado", fc_bpm=0, marcapasso=False)
    r = compare(sistema, gt, case_id=1)
    assert _cat(r, "marcapasso") == "CATCH"
    assert _cat(r, "ritmo") == "CATCH"


def test_clinical_disagreement_detection():
    sistema = ECGFindings(
        ritmo="sinusal", fc_bpm=80,
        st_supra=STDesvio(derivacoes=["V2", "V3"], magnitude_mm_max=2.0),
    )
    gt = ECGFindings(ritmo="sinusal", fc_bpm=80)
    r = compare(sistema, gt, case_id=1)
    assert r.has_clinical_disagreement()


def test_no_clinical_disagreement_on_match():
    f = ECGFindings(ritmo="sinusal", fc_bpm=75)
    r = compare(f, f, case_id=1)
    assert not r.has_clinical_disagreement()


def test_qtc_catch():
    sistema = ECGFindings(ritmo="sinusal", fc_bpm=80, qtc_ms=510)
    gt = ECGFindings(ritmo="sinusal", fc_bpm=80, qtc_ms=0)
    r = compare(sistema, gt, case_id=1)
    qtc_items = [i for i in r.itens if i.campo == "qtc"]
    assert qtc_items[0].categoria == "CATCH"
