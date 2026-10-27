"""Gera planilha para a médica classificar os CATCHes e HALLUCINATIONs.

Uso:
  python -m eval.review_sheet                  # stdout CSV
  python -m eval.review_sheet -o review.csv    # arquivo
  python -m eval.review_sheet --import done.csv  # importa veredito"""

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage import db


CSV_COLUMNS = [
    "review_id", "case_id", "campo",
    "sistema", "gt",
    "caso_clinico", "laudo_gt",
    "veredito",  # VERDADEIRO | FALSO | IRRELEVANTE
    "observacao",
]


def export(output_path: str | None) -> None:
    reviews = db.list_catch_reviews()
    if not reviews:
        sys.stderr.write("Sem catch_reviews. Rode python -m eval.runner antes.\n")
        return

    case_cache: dict[int, db.sqlite3.Row] = {}
    rows = []
    for r in reviews:
        cid = r["case_id"]
        if cid not in case_cache:
            case_cache[cid] = db.get_eval_case(cid)
        cs = case_cache[cid]
        rows.append({
            "review_id": r["id"],
            "case_id": cid,
            "campo": r["campo"],
            "sistema": r["sistema"],
            "gt": r["gt"] or "—",
            "caso_clinico": cs["caso_texto"] if cs else "",
            "laudo_gt": cs["laudo_gt"] if cs else "",
            "veredito": r["veredito"] or "",
            "observacao": r["observacao"] or "",
        })

    out = sys.stdout if not output_path else open(output_path, "w", encoding="utf-8")
    w = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
    w.writeheader()
    w.writerows(rows)
    if output_path:
        out.close()
        print(f"✓ {len(rows)} linhas em {output_path}")


def import_verdict(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        updated = 0
        for row in reader:
            veredito = (row.get("veredito") or "").strip().lower()
            if veredito not in ("verdadeiro", "falso", "irrelevante"):
                continue
            db.set_catch_verdict(
                review_id=int(row["review_id"]),
                veredito=veredito,
                observacao=(row.get("observacao") or "").strip() or None,
            )
            updated += 1
    print(f"✓ {updated} veredictos atualizados")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=None, help="caminho CSV de saída")
    parser.add_argument("--import", dest="import_path", default=None,
                        help="caminho CSV com vereditos preenchidos")
    args = parser.parse_args()

    db.init_db()
    if args.import_path:
        import_verdict(args.import_path)
    else:
        export(args.output)


if __name__ == "__main__":
    main()
