"""Compara os dois braços do A/B (preprocess ON vs OFF), pareado por uid.

Lê results/arm_on.jsonl e results/arm_off.jsonl e imprime:
  - totais por métrica em cada braço
  - delta pareado (só casos ok nos DOIS braços)
  - contagem de casos onde ON ficou melhor / pior / igual

Score clínico por caso = AGREE - MISS - 2*HALLUC
  (acordo é bom; perder achado do GT é ruim; alucinar é o pior — pesa 2x).
  CATCH é ignorado no score: GT telegráfico, "sistema viu e GT não" é ruidoso.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(arm):
    p = RESULTS / f"arm_{arm}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def score(c):
    return c["agree"] - c["miss"] - 2 * c["halluc"]


def main():
    on, off = load("on"), load("off")
    print(f"arm ON : {len(on)} registros   ({sum(1 for r in on.values() if r['ok'])} ok)")
    print(f"arm OFF: {len(off)} registros   ({sum(1 for r in off.values() if r['ok'])} ok)")

    paired = [u for u in on if u in off and on[u]["ok"] and off[u]["ok"]]
    print(f"\nPareados ok nos dois braços: {len(paired)}")
    if not paired:
        print("Sem pares — rode os dois braços primeiro.")
        return

    METS = ("agree", "miss", "halluc", "catch")
    tot_on = {m: sum(on[u]["counters"][m] for u in paired) for m in METS}
    tot_off = {m: sum(off[u]["counters"][m] for u in paired) for m in METS}

    print("\n" + "=" * 46)
    print(f"{'métrica':<10}{'OFF':>8}{'ON':>8}{'Δ(ON-OFF)':>14}")
    print("-" * 46)
    for m in METS:
        d = tot_on[m] - tot_off[m]
        print(f"{m:<10}{tot_off[m]:>8}{tot_on[m]:>8}{d:>+14}")

    s_on = sum(score(on[u]["counters"]) for u in paired)
    s_off = sum(score(off[u]["counters"]) for u in paired)
    print("-" * 46)
    print(f"{'SCORE':<10}{s_off:>8}{s_on:>8}{s_on - s_off:>+14}")
    print("=" * 46)

    better = worse = equal = 0
    for u in paired:
        d = score(on[u]["counters"]) - score(off[u]["counters"])
        if d > 0:
            better += 1
        elif d < 0:
            worse += 1
        else:
            equal += 1
    print(f"\nPor caso (score): ON melhor={better}  pior={worse}  igual={equal}")

    verdict = ("PREPROCESS AJUDA" if s_on > s_off else
               "PREPROCESS ATRAPALHA" if s_on < s_off else "EMPATE")
    print(f"\n>>> {verdict} (score {s_off} → {s_on})")

    # casos com maior regressão/ganho — pra inspeção visual
    deltas = sorted(paired, key=lambda u: score(on[u]["counters"]) - score(off[u]["counters"]))
    print("\nMaiores REGRESSÕES do ON (inspecionar imagem):")
    for u in deltas[:5]:
        d = score(on[u]["counters"]) - score(off[u]["counters"])
        if d < 0:
            print(f"  {u:>14}  Δ={d:+d}  off={off[u]['counters']}  on={on[u]['counters']}")
    print("Maiores GANHOS do ON:")
    for u in deltas[::-1][:5]:
        d = score(on[u]["counters"]) - score(off[u]["counters"])
        if d > 0:
            print(f"  {u:>14}  Δ={d:+d}  off={off[u]['counters']}  on={on[u]['counters']}")


if __name__ == "__main__":
    main()
