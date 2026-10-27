"""Seleciona os 10 casos do piloto a partir dos resultados do A/B:
  - 5 que os modelos MAIS ERRARAM (menor score somado ON+OFF)
  - 5 aleatórios (seed fixa=42) do restante

score por caso = AGREE - MISS - 2*HALLUC  (menor = pior)
Só considera casos ok em AMBOS os braços (têm score confiável) e com imagem.
Imprime os 10 uids (vírgula) p/ passar ao pilot_fewshot.py.
"""

import json
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(arm):
    p = RESULTS / f"arm_{arm}.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    out[r["uid"]] = r["counters"]
    return out


def score(c):
    return c["agree"] - c["miss"] - 2 * c["halluc"]


def main():
    off, on = load("off"), load("on")
    both = [u for u in off if u in on]
    combined = {u: score(off[u]) + score(on[u]) for u in both}
    ranked = sorted(both, key=lambda u: combined[u])   # pior primeiro

    worst5 = ranked[:5]
    rest = [u for u in ranked if u not in worst5]
    random.seed(42)
    rand5 = random.sample(rest, min(5, len(rest)))

    print(f"# casos ok nos dois braços: {len(both)}")
    print("# 5 PIORES (score somado ON+OFF):")
    for u in worst5:
        print(f"#   {u}  score={combined[u]}  off={off[u]} on={on[u]}")
    print("# 5 ALEATÓRIOS (seed=42):")
    for u in rand5:
        print(f"#   {u}  score={combined[u]}")
    sel = worst5 + rand5
    print("\n" + ",".join(sel))
    (RESULTS / "pilot_cases.txt").write_text(",".join(sel))


if __name__ == "__main__":
    main()
