"""Per-finding evaluation of the vision step against the physician's report.

Reads the private outputs of experiments/ab_preprocess.py:
  experiments/results/arm_on.jsonl, arm_off.jsonl   model findings per case
  experiments/results/gt_cache.json                 physician report parsed into the same schema

and writes results/aggregate_metrics.json-style counts (TP/FN/FP/TN per finding),
plus sensitivity / specificity with Wilson 95% intervals.

Reading the numbers: the reference is a short free-text report. When a physician
mentions a finding it is almost certainly there, so sensitivity is meaningful.
When a finding is not mentioned it may still be on the tracing, so "false
positives" (and specificity) are an upper bound on the real over-calling.

Usage:
  python analysis/finding_metrics.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"

FLAGS = {
    "atrial_fibrillation_flutter": lambda f: f["ritmo"] in ("fa", "flutter_atrial")
    or "fibriloflutter" in f["achados_especiais"],
    "non_sinus_rhythm": lambda f: f["ritmo"] not in ("sinusal", "indeterminado"),
    "st_elevation": lambda f: len(f["st_supra"]["derivacoes"]) > 0,
    "st_depression": lambda f: len(f["st_infra"]["derivacoes"]) > 0,
    "lbbb": lambda f: "bre" in f["bloqueios"],
    "rbbb": lambda f: "brd" in f["bloqueios"],
    "av_block_any": lambda f: any(b.startswith("bav") for b in f["bloqueios"]),
    "fascicular_block": lambda f: any(b in ("bdas", "bdpi") for b in f["bloqueios"]),
    "lvh": lambda f: bool(f["hipertrofia_ve"]),
    "pacemaker": lambda f: bool(f["marcapasso"]) or f["ritmo"] == "marcapasso",
    "q_waves_inactive_area": lambda f: len(f["areas_inativas"]) > 0,
    "t_wave_changes": lambda f: len(f["onda_t_alteracoes"]) > 0,
    "any_abnormality": lambda f: not (
        f["ritmo"] == "sinusal" and not f["bloqueios"]
        and not f["st_supra"]["derivacoes"] and not f["st_infra"]["derivacoes"]
        and not f["onda_t_alteracoes"] and not f["areas_inativas"]
        and not f["hipertrofia_ve"] and not f["hipertrofia_vd"]
        and not f["marcapasso"] and not f["achados_especiais"]
    ),
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_arm(name: str) -> dict:
    out = {}
    for line in (RES / f"arm_{name}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok") and r.get("findings"):
                out[r["uid"]] = r["findings"]
    return out


def evaluate(pred: dict, gt: dict) -> dict:
    uids = [u for u in pred if u in gt]
    table = {}
    for name, fn in FLAGS.items():
        tp = fn_ = fp = tn = 0
        for u in uids:
            g, p = fn(gt[u]), fn(pred[u])
            tp += g and p
            fn_ += g and not p
            fp += p and not g
            tn += (not g) and (not p)
        table[name] = {"tp": tp, "fn": fn_, "fp": fp, "tn": tn}
    return table


def main():
    raw = json.loads((RES / "gt_cache.json").read_text())
    gt = {k: json.loads(v) if isinstance(v, str) else v for k, v in raw.items()}
    for arm in ("on", "off"):
        t = evaluate(load_arm(arm), gt)
        print(f"\npreprocess {arm.upper()}")
        print(f"{'finding':28}{'ref+':>5}{'sens':>6}{'95% CI':>14}{'spec':>7}")
        for name, c in t.items():
            s = wilson(c["tp"], c["tp"] + c["fn"])
            sp = wilson(c["tn"], c["tn"] + c["fp"])
            print(f"{name:28}{c['tp'] + c['fn']:>5}{s[0]:>6.2f}  [{s[1]:.2f}, {s[2]:.2f}]{sp[0]:>7.2f}")
        print(json.dumps(t))


if __name__ == "__main__":
    main()
