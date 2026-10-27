"""Amostra estratificada para validação do juiz pela médica.

Pega o veredito mais recente por caso (Haiku v2), estratifica por bucket
(PERFEITO/MEIO_LAUDO/MEIO_CONDUTA/ZERO) e exporta CSV.

Uso:
  python -m eval.medica_review --output /tmp/medica_review.csv
  python -m eval.medica_review --import /tmp/medica_review.csv  # após preenchida"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage import db

random.seed(42)


def _bucket(veredito_l: str, veredito_c: str) -> str:
    la = veredito_l in ("ACEITARIA", "ACEITARIA_COM_EDICAO")
    ca = veredito_c in ("ACEITARIA", "ACEITARIA_COM_EDICAO")
    if la and ca:
        return "PERFEITO"
    if la:
        return "MEIO_LAUDO"
    if ca:
        return "MEIO_CONDUTA"
    return "ZERO"


def export(output_path: str):
    with db.connect() as conn:
        rows = list(conn.execute("""
            SELECT j.id AS judgment_id, j.case_id,
                   j.veredito_laudo, j.motivo_laudo,
                   j.veredito_conduta, j.motivo_conduta,
                   c.caso_texto, c.laudo_gt, c.conduta_gt,
                   r.laudo AS laudo_sistema, r.conduta AS conduta_sistema
            FROM acceptance_judgments j
            JOIN eval_cases c ON c.id = j.case_id
            JOIN runs r ON r.id = j.run_id
            WHERE j.id IN (SELECT MAX(id) FROM acceptance_judgments GROUP BY case_id)
            ORDER BY j.case_id
        """))

    if not rows:
        sys.exit("Sem judgments. Rode python -m eval.acceptance --run")

    # Estratifica
    buckets = defaultdict(list)
    for r in rows:
        b = _bucket(r["veredito_laudo"], r["veredito_conduta"])
        buckets[b].append(r)

    print("Distribuição:")
    for b in ("PERFEITO", "MEIO_LAUDO", "MEIO_CONDUTA", "ZERO"):
        print(f"  {b:14s} {len(buckets[b]):3d}")

    # Amostragem alvo: 20 casos
    targets = {"ZERO": 7, "MEIO_CONDUTA": 5, "MEIO_LAUDO": 3, "PERFEITO": 5}
    sample = []
    for b, n in targets.items():
        pool = buckets[b]
        if not pool:
            continue
        random.shuffle(pool)
        sample.extend(pool[:n])

    # Ordena: ZERO primeiro (mais críticos), depois MEIO_CONDUTA, MEIO_LAUDO, PERFEITO
    order = {"ZERO": 0, "MEIO_CONDUTA": 1, "MEIO_LAUDO": 2, "PERFEITO": 3}
    sample.sort(key=lambda r: (order[_bucket(r["veredito_laudo"], r["veredito_conduta"])],
                                r["case_id"]))

    cols = [
        "ordem",
        "case_id",
        "bucket_juiz",
        "caso_clinico",
        "laudo_cardiologista",
        "laudo_sistema",
        "veredito_juiz_laudo",
        "motivo_juiz_laudo",
        "MEDICA_concorda_laudo",  # SIM/NAO
        "MEDICA_veredito_laudo",  # ACEITARIA / ACEITARIA_COM_EDICAO / REJEITARIA
        "MEDICA_obs_laudo",
        "conduta_cardiologista",
        "conduta_sistema",
        "veredito_juiz_conduta",
        "motivo_juiz_conduta",
        "MEDICA_concorda_conduta",  # SIM/NAO
        "MEDICA_veredito_conduta",
        "MEDICA_obs_conduta",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow({
                "ordem": i,
                "case_id": r["case_id"],
                "bucket_juiz": _bucket(r["veredito_laudo"], r["veredito_conduta"]),
                "caso_clinico": r["caso_texto"],
                "laudo_cardiologista": r["laudo_gt"],
                "laudo_sistema": r["laudo_sistema"] or "",
                "veredito_juiz_laudo": r["veredito_laudo"],
                "motivo_juiz_laudo": r["motivo_laudo"] or "",
                "MEDICA_concorda_laudo": "",
                "MEDICA_veredito_laudo": "",
                "MEDICA_obs_laudo": "",
                "conduta_cardiologista": r["conduta_gt"],
                "conduta_sistema": r["conduta_sistema"] or "",
                "veredito_juiz_conduta": r["veredito_conduta"],
                "motivo_juiz_conduta": r["motivo_conduta"] or "",
                "MEDICA_concorda_conduta": "",
                "MEDICA_veredito_conduta": "",
                "MEDICA_obs_conduta": "",
            })

    print(f"\n✓ {len(sample)} casos exportados em {output_path}")
    print()
    print("Instruções para a médica:")
    print("  1. Abrir o CSV no Excel / Google Sheets")
    print("  2. Para cada linha, comparar:")
    print("     - laudo_cardiologista vs laudo_sistema")
    print("     - conduta_cardiologista vs conduta_sistema")
    print("  3. Preencher:")
    print("     - MEDICA_concorda_laudo: SIM (concorda com o juiz) ou NAO")
    print("     - MEDICA_veredito_laudo: o que ela mesma daria como veredito")
    print("       (ACEITARIA / ACEITARIA_COM_EDICAO / REJEITARIA)")
    print("     - MEDICA_obs_laudo: por que (livre, opcional)")
    print("     - mesmas 3 colunas para conduta")
    print()
    print("Calibração do juiz =")
    print("    nº de linhas com MEDICA_concorda_laudo=SIM / 20")
    print("    nº de linhas com MEDICA_concorda_conduta=SIM / 20")
    print("Meta: ≥85% de concordância em ambos.")


def import_back(path: str):
    """Importa o CSV preenchido pela médica e calcula concordância."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    concorda_l = sum(1 for r in rows if (r.get("MEDICA_concorda_laudo") or "").strip().upper() == "SIM")
    concorda_c = sum(1 for r in rows if (r.get("MEDICA_concorda_conduta") or "").strip().upper() == "SIM")

    # Persistir
    for r in rows:
        try:
            jid = int(r["case_id"])
        except (ValueError, KeyError):
            continue
        with db.connect() as conn:
            conn.execute(
                """UPDATE acceptance_judgments
                   SET medico_veredito_laudo=?, medico_veredito_conduta=?, medico_obs=?
                   WHERE case_id=? AND id=(SELECT MAX(id) FROM acceptance_judgments WHERE case_id=?)""",
                (
                    (r.get("MEDICA_veredito_laudo") or "").strip() or None,
                    (r.get("MEDICA_veredito_conduta") or "").strip() or None,
                    ((r.get("MEDICA_obs_laudo") or "") + " || " + (r.get("MEDICA_obs_conduta") or "")).strip(" |") or None,
                    jid, jid,
                ),
            )

    print(f"═══ Calibração do juiz em {n} casos ═══")
    print(f"  Médica concorda c/ juiz no LAUDO   : {concorda_l}/{n}  ({100*concorda_l/n:.0f}%)")
    print(f"  Médica concorda c/ juiz na CONDUTA : {concorda_c}/{n}  ({100*concorda_c/n:.0f}%)")
    if concorda_l/n >= 0.85 and concorda_c/n >= 0.85:
        print("\n  ✓ Juiz CALIBRADO (≥85% em ambos). Números do report são confiáveis.")
    else:
        print("\n  ⚠️ Juiz NÃO calibrado. Revisar critérios + reajustar prompt.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=None, help="caminho para gerar o CSV")
    p.add_argument("--import-path", default=None, help="caminho com CSV preenchido")
    args = p.parse_args()

    if args.output:
        export(args.output)
    elif args.import_path:
        import_back(args.import_path)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
