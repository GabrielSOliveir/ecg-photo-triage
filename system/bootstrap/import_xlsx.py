"""Importa os 55 casos do xlsx para a tabela eval_cases.

Uso: python -m bootstrap.import_xlsx

Não copia imagens — apenas armazena o caminho absoluto. As imagens permanecem
no diretório do v1 (Imagens_ECG/). Casos sem imagem correspondente são pulados."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openpyxl

from config import EVAL_XLSX_PATH, EVAL_IMAGES_DIR
from storage import db


def _find_image(case_id: int) -> Path | None:
    for ext in (".jpeg", ".jpg", ".png", ".webp"):
        p = EVAL_IMAGES_DIR / f"{case_id}{ext}"
        if p.exists():
            return p
    return None


def main():
    db.init_db()
    wb = openpyxl.load_workbook(EVAL_XLSX_PATH)
    ws = wb.active

    imported = 0
    skipped = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        cid = int(row[0])
        caso_texto = (row[1] or "").strip()
        laudo_gt = (row[2] or "").strip()
        conduta_gt = (row[3] or "").strip()
        data_str = str(row[4]) if row[4] else None

        if not caso_texto or not laudo_gt:
            skipped.append((cid, "sem texto ou laudo"))
            continue

        img_path = _find_image(cid)
        if not img_path:
            skipped.append((cid, "sem imagem"))
            continue

        db.import_eval_case(
            case_id=cid,
            caso_texto=caso_texto,
            laudo_gt=laudo_gt,
            conduta_gt=conduta_gt,
            data_original=data_str,
            imagem_path=str(img_path),
        )
        imported += 1

    print(f"✓ Importados: {imported} casos")
    if skipped:
        print(f"✗ Pulados: {len(skipped)}")
        for cid, motivo in skipped:
            print(f"  - {cid}: {motivo}")


if __name__ == "__main__":
    main()
