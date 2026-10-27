"""v3 — A/B pareado de PRÉ-PROCESSAMENTO de imagem na etapa de vision.

Pergunta única: o preprocess clássico da v2 (deskew + crop papel + CLAHE)
melhora os *findings* extraídos, comparado a mandar a foto crua?

Desenho:
  - Mesmos casos, mesmo modelo (Claude via system/.env), mesmo prompt.
  - Único eixo que muda: PREPROCESS_VISION on/off.
  - GT (laudo da médica) é parseado UMA vez e cacheado → idêntico nos 2 braços,
    então qualquer erro do parser de GT cancela no comparativo pareado.
  - Reusa as peças de lógica da v2: vision.extractor, gt.parser, eval.comparator.

Uso:
  python ab_preprocess.py --arm on  --limit 15
  python ab_preprocess.py --arm off --limit 15
  python summarize.py

IMPORTANTE: os flags de preprocess são lidos do ambiente no import do config.
Por isso setamos os.environ ANTES de importar qualquer coisa da v2, e rodamos
um processo por braço (garante estado limpo).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── resolve o braço ANTES de importar config/v2 ──────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--arm", choices=["on", "off"], required=True)
_parser.add_argument("--limit", type=int, default=0, help="0 = todos")
_parser.add_argument("--only", type=str, default="", help="IDs separados por vírgula")
_parser.add_argument("--workers", type=int, default=4)
_args = _parser.parse_args()

_ON = _args.arm == "on"
os.environ["PREPROCESS_VISION"] = "1" if _ON else "0"
# no braço ON usamos exatamente o pipeline default da v2 (deskew+crop+clahe)
os.environ.setdefault("PREPROCESS_DESKEW", "1")
os.environ.setdefault("PREPROCESS_CROP_PAPER", "1")
os.environ.setdefault("PREPROCESS_CLAHE", "1")

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "system"
sys.path.insert(0, str(V2))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

from config import VISION_MODEL  # noqa: E402
from schemas import PatientData, ECGFindings  # noqa: E402
from vision.extractor import extract_findings, VisionExtractionError  # noqa: E402
from gt.parser import parse_gt_laudo, GTParseError  # noqa: E402
from eval.comparator import compare  # noqa: E402

XLSX = ROOT / "dataset" / "casos_unificados.xlsx"
IMAGES = ROOT / "imagens"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
GT_CACHE = RESULTS / "gt_cache.json"
OUT = RESULTS / f"arm_{'on' if _ON else 'off'}.jsonl"


def load_cases():
    ws = openpyxl.load_workbook(XLSX, read_only=True).active
    header = None
    rows = []
    for r in ws.iter_rows(values_only=True):
        if header is None:
            header = list(r)
            continue
        rows.append(dict(zip(header, r)))
    cases = []
    for r in rows:
        img = r.get("imagem")
        laudo = r.get("laudo")
        if not img or not laudo:
            continue
        p = IMAGES / str(img)
        if not p.exists():
            continue
        origem = str(r.get("origem") or "?")
        cases.append({
            "uid": f"{origem}:{r['id']}",   # id NÃO é único entre origens
            "idx": len(cases),               # inteiro estável p/ o comparator (id pode ser string)
            "id": r["id"],
            "origem": origem,
            "caso_texto": str(r.get("caso_texto") or ""),
            "laudo_gt": str(laudo),
            "img": p,
        })
    return cases


def to_jpeg_bytes(path: Path) -> bytes:
    """Baseline comum aos 2 braços: decodifica qualquer formato (png/jpeg) e
    re-encoda JPEG q95. Garante media_type=jpeg (o extractor da v2 assume jpeg)
    e mesmo ponto de partida — o único eixo que varia é o preprocess."""
    arr = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return path.read_bytes()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes() if ok else path.read_bytes()


# ── cache de GT (parse do laudo → ECGFindings), compartilhado entre braços ────
def load_gt_cache() -> dict:
    if GT_CACHE.exists():
        return json.loads(GT_CACHE.read_text())
    return {}


def get_gt(case, cache) -> ECGFindings:
    key = case["uid"]
    if key in cache:
        return ECGFindings.model_validate_json(cache[key])
    parsed = parse_gt_laudo(case["laudo_gt"])
    cache[key] = parsed.model_dump_json()
    GT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    return parsed


def run_case(case, gt_cache) -> dict:
    cid = case["id"]
    base = {"uid": case["uid"], "id": str(cid)}
    t0 = time.time()
    patient = PatientData(queixa=case["caso_texto"], contexto=case["caso_texto"])
    try:
        img_bytes = to_jpeg_bytes(case["img"])   # baseline jpeg idêntico p/ ambos
        findings = extract_findings(img_bytes, patient, model=VISION_MODEL)
    except Exception as e:
        return {**base, "ok": False, "erro": f"vision: {e}"}
    try:
        gt = get_gt(case, gt_cache)
    except Exception as e:
        return {**base, "ok": False, "erro": f"gt: {e}"}
    try:
        cmp = compare(findings, gt, case_id=case["idx"])   # idx inteiro (id pode ser string)
    except Exception as e:
        return {**base, "ok": False, "erro": f"compare: {e}"}
    return {
        "uid": case["uid"],
        "id": str(cid),
        "ok": True,
        "elapsed": round(time.time() - t0, 1),
        "counters": {
            "agree": cmp.n_agree,
            "catch": cmp.n_catch,
            "miss": cmp.n_miss,
            "halluc": cmp.n_hallucination,
            "neutral": cmp.n_neutral_disagree,
        },
        "findings": findings.model_dump(),
    }


def main():
    cases = load_cases()
    if _args.only:
        ids = {int(x) for x in _args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in ids]
    if _args.limit:
        cases = cases[: _args.limit]

    print(f"[arm={_args.arm}] modelo={VISION_MODEL} PREPROCESS_VISION={os.environ['PREPROCESS_VISION']}")
    print(f"[arm={_args.arm}] {len(cases)} casos → {OUT.name}")

    gt_cache = load_gt_cache()
    # pré-popula GT em série (evita corrida de escrita no json entre threads)
    for c in cases:
        try:
            get_gt(c, gt_cache)
        except GTParseError as e:
            print(f"   [gt-warn] caso {c['id']}: {e}")

    import threading
    results = []
    write_lock = threading.Lock()
    OUT.write_text("")   # zera; escrevemos incremental (crash não perde o lote)
    with ThreadPoolExecutor(max_workers=_args.workers) as ex:
        futs = {ex.submit(run_case, c, gt_cache): c for c in cases}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            with write_lock, OUT.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["ok"]:
                cc = r["counters"]
                print(f"   caso {r['id']:>6}: AGREE={cc['agree']} MISS={cc['miss']} "
                      f"HALLUC={cc['halluc']} CATCH={cc['catch']} ({r['elapsed']}s)", flush=True)
            else:
                print(f"   caso {r['id']:>6}: ERRO {r['erro']}", flush=True)

    ok = [r for r in results if r["ok"]]
    if ok:
        agg = {k: sum(r["counters"][k] for r in ok) for k in ("agree", "miss", "halluc", "catch", "neutral")}
        print(f"\n[arm={_args.arm}] {len(ok)}/{len(results)} ok | TOTAIS {agg}")


if __name__ == "__main__":
    main()
