"""Builds the README figures from results/aggregate_metrics.json (aggregate counts only)."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "results" / "aggregate_metrics.json").read_text())
OUT = Path(__file__).resolve().parent

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

LABELS = {
    "any_abnormality": "Any abnormality",
    "t_wave_changes": "T-wave changes",
    "st_depression": "ST depression",
    "lvh": "Left ventricular hypertrophy",
    "atrial_fibrillation_flutter": "Atrial fibrillation / flutter",
    "non_sinus_rhythm": "Non-sinus rhythm",
    "rbbb": "Right bundle branch block",
    "av_block_any": "AV block",
    "st_elevation": "ST elevation",
    "q_waves_inactive_area": "Pathological Q waves",
    "lbbb": "Left bundle branch block",
    "fascicular_block": "Fascicular block",
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def fig_sensitivity():
    on, off = M["per_finding"]["preprocess_on"], M["per_finding"]["preprocess_off"]
    keys = [k for k in LABELS if on[k]["tp"] + on[k]["fn"] >= 5]
    keys.sort(key=lambda k: on[k]["tp"] / (on[k]["tp"] + on[k]["fn"]))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for i, k in enumerate(keys):
        n_on = on[k]["tp"] + on[k]["fn"]
        p, lo, hi = wilson(on[k]["tp"], n_on)
        p_off = off[k]["tp"] / (off[k]["tp"] + off[k]["fn"])
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2, alpha=0.35, solid_capstyle="round")
        ax.plot([p_off, p], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(p_off, i, s=46, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.scatter(p, i, s=64, color=BLUE, zorder=4, edgecolor=SURFACE, linewidth=2)
        ax.text(1.02, i, f"{on[k]['tp']}/{n_on}", va="center", fontsize=9, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(keys)), [LABELS[k] for k in keys], color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sensitivity vs. the physician's report (shaded: 95% CI, preprocessed photo)")
    ax.scatter([], [], s=64, color=BLUE, label="Preprocessed photo (deskew + crop + CLAHE)")
    ax.scatter([], [], s=46, color=ORANGE, label="Raw photo")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
              handletextpad=0.3, columnspacing=1.5)
    fig.suptitle("Findings the model caught, out of those the physician reported (225 photos)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.text(0.99, 0.01, "right column: found / reported", ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "01_sensitivity_by_finding.png", dpi=170)
    plt.close(fig)


def fig_production():
    p = M["production"]
    rows = [("Report (laudo)", p["report_similarity_per_case"]),
            ("Plan (conduta)", p["plan_similarity_per_case"])]
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    for i, (name, vals) in enumerate(rows):
        ys = [i + ((j % 5) - 2) * 0.07 for j in range(len(vals))]
        ax.scatter(vals, ys, s=52, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        med = sorted(vals)[len(vals) // 2]
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.text(med, i + 0.36, f"median {med:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_yticks([0, 1], [r[0] for r in rows], color=INK)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(-0.02, 1.04)
    ax.set_xticks([0, .25, .5, .75, 1], ["0\nrewritten", "0.25", "0.5", "0.75", "1\nunchanged"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Production, 21 cases: how much of the model's text the physician kept",
                 loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "02_production_edits.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_sensitivity()
    fig_production()
