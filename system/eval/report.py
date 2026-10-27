"""Relatório agregado do eval. Lê SQLite e responde à pergunta-chave:
'Em quantos casos o sistema teria pego algo que passou batido, e em quantos teria errado?'

Uso: python -m eval.report"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage import db


def main():
    db.init_db()
    comps = db.list_comparisons()
    if not comps:
        print("Sem comparisons. Rode: python -m eval.runner")
        return

    n_cases = len(comps)
    cat_counter = Counter()
    catch_cases: list[int] = []
    miss_cases: list[int] = []
    halluc_cases: list[int] = []
    perfeitos: list[int] = []

    per_field: dict[str, Counter] = {}

    for c in comps:
        result = json.loads(c["result_json"])
        items = result.get("itens", [])
        has_catch = False
        has_miss = False
        has_halluc = False
        for it in items:
            if it["peso_clinico"] not in ("medio", "alto"):
                continue
            cat_counter[it["categoria"]] += 1
            per_field.setdefault(it["campo"], Counter())[it["categoria"]] += 1
            if it["categoria"] == "CATCH":
                has_catch = True
            elif it["categoria"] == "MISS":
                has_miss = True
            elif it["categoria"] == "HALLUCINATION":
                has_halluc = True
        if has_catch:
            catch_cases.append(c["case_id"])
        if has_miss:
            miss_cases.append(c["case_id"])
        if has_halluc:
            halluc_cases.append(c["case_id"])
        if not (has_catch or has_miss or has_halluc):
            perfeitos.append(c["case_id"])

    print("=" * 60)
    print(f"RELATÓRIO DE EVAL — {n_cases} casos comparados")
    print("=" * 60)
    print()
    print("Resposta à pergunta-chave (filtrando peso clínico médio/alto):")
    print(f"  → CATCH (sistema pegou, GT não): {len(catch_cases)} casos")
    print(f"     IDs: {sorted(catch_cases)}")
    print(f"  → MISS (GT pegou, sistema não): {len(miss_cases)} casos")
    print(f"     IDs: {sorted(miss_cases)}")
    print(f"  → HALLUCINATION (sistema afirma X, GT afirma Y): {len(halluc_cases)} casos")
    print(f"     IDs: {sorted(halluc_cases)}")
    print(f"  → Sem discordância clínica: {len(perfeitos)} casos")
    print()
    print(f"Razão CATCH/(MISS+HALLUC) = {len(catch_cases)} / {len(miss_cases) + len(halluc_cases)}")
    if len(miss_cases) + len(halluc_cases) > 0:
        ratio = len(catch_cases) / (len(miss_cases) + len(halluc_cases))
        print(f"  → Benefício / dano: {ratio:.2f}  ({'COMPENSA' if ratio >= 1 else 'NÃO COMPENSA AINDA'})")
    print()
    print("Discordâncias individuais (peso médio/alto):")
    print(f"  AGREE: {cat_counter['AGREE']}")
    print(f"  CATCH: {cat_counter['CATCH']}")
    print(f"  MISS: {cat_counter['MISS']}")
    print(f"  HALLUCINATION: {cat_counter['HALLUCINATION']}")
    print()
    print("Por campo:")
    for campo, counter in sorted(per_field.items()):
        line = f"  {campo:20s} "
        for cat in ("AGREE", "CATCH", "MISS", "HALLUCINATION"):
            line += f"{cat}={counter[cat]:3d}  "
        print(line)
    print()
    print("⚠️  IMPORTANTE: CATCH e HALLUCINATION são *candidatos*. Para")
    print("    o número final, a médica precisa revisar via review_sheet:")
    print("    python -m eval.review_sheet > catch_review.csv")
    print()
    print("    Após revisão (preencher veredito no CSV e re-importar),")
    print("    os CATCH viram VERDADEIROS (sistema agregou valor) vs FALSOS.")


if __name__ == "__main__":
    main()
