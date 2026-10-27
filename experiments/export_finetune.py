"""Exporta os laudos reais das médicas → JSONL de fine-tuning de visão da OpenAI.

Fonte PRIMÁRIA: dataset/casos_unificados.xlsx — ~228 pares reais
    (imagem do ECG + laudo escrito por médica + contexto clínico).
Fonte de REFORÇO (opcional, --com-producao): correções novas do bot
    (doctor_decisions.laudo_final) que ainda não estão no xlsx.

Cada exemplo:
    system   = instruções de laudo
    user     = contexto clínico (caso_texto) + IMAGEM
    assistant= laudo da médica  (o alvo)

O modelo aprende imagem → laudo direto, internalizando a voz da médica.

Uso:
  python export_finetune.py                 # só os 228 do xlsx
  python export_finetune.py --com-producao  # + correções do bot não duplicadas
  python export_finetune.py --split 0.85    # separa treino/val
"""

import argparse
import base64
import json
import sqlite3
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "dataset" / "casos_unificados.xlsx"
IMAGES = ROOT / "imagens"
V2 = ROOT / "system"
DB = V2 / "data" / "ecg.db"
PROD_IMG = V2 / "data" / "images" / "production"
OUTDIR = Path(__file__).resolve().parent / "results"

SYSTEM = ("Você é cardiologista experiente. A partir da foto do ECG e dos dados "
          "clínicos do paciente, escreva o laudo eletrocardiográfico no estilo de "
          "prontuário deste serviço: 2-4 frases descritivas dos achados.")


def img_data_url(path: Path) -> str:
    b = path.read_bytes()
    mt = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mt};base64," + base64.standard_b64encode(b).decode()


def example(caso_texto: str, img: Path, laudo: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": (caso_texto or "").strip() or "(sem contexto)"},
            {"type": "image_url", "image_url": {"url": img_data_url(img)}},
        ]},
        {"role": "assistant", "content": laudo.strip()},
    ]}


def from_xlsx():
    ws = openpyxl.load_workbook(XLSX, read_only=True).active
    header, exs, seen = None, [], set()
    for r in ws.iter_rows(values_only=True):
        if header is None:
            header = list(r); continue
        row = dict(zip(header, r))
        img, laudo = row.get("imagem"), row.get("laudo")
        if not img or not laudo:
            continue
        p = IMAGES / str(img)
        if not p.exists():
            continue
        exs.append(example(str(row.get("caso_texto") or ""), p, str(laudo)))
        seen.add(f"{row.get('origem')}:{row.get('id')}")
    return exs, seen


def from_producao(seen):
    if not DB.exists():
        return []
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    exs = []
    for r in c.execute("""
        SELECT dd.laudo_final, r.laudo AS laudo_modelo, dd.decision,
               pc.caso_texto, pc.imagem_path
        FROM doctor_decisions dd
        JOIN production_cases pc ON pc.id = dd.production_case_id
        LEFT JOIN runs r ON r.id = dd.run_id
    """):
        target = (r["laudo_final"] or (r["laudo_modelo"] if r["decision"] == "aprovado" else "") or "").strip()
        if not target:
            continue
        p = Path(r["imagem_path"])
        cand = next((x for x in (p, PROD_IMG / p.name, V2 / r["imagem_path"]) if x.exists()), None)
        if not cand:
            continue
        exs.append(example(r["caso_texto"] or "", cand, target))
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--com-producao", action="store_true")
    ap.add_argument("--split", type=float, default=0.0, help="fração de treino (resto=val)")
    args = ap.parse_args()

    exs, seen = from_xlsx()
    print(f"xlsx (laudos reais): {len(exs)}")
    if args.com_producao:
        prod = from_producao(seen)
        print(f"produção (correções do bot): +{len(prod)}")
        exs += prod

    if args.split and 0 < args.split < 1:
        n = int(len(exs) * args.split)
        (OUTDIR / "finetune_train.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in exs[:n]))
        (OUTDIR / "finetune_val.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in exs[n:]))
        print(f"train={n} → finetune_train.jsonl | val={len(exs)-n} → finetune_val.jsonl")
    else:
        out = OUTDIR / "finetune_openai.jsonl"
        out.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in exs))
        print(f"total={len(exs)} → {out.name}")

    # estimativa de custo de treino (imagem ~1.1k tokens + laudo ~200)
    tok = len(exs) * 1300
    print(f"\n~{tok} tokens/época. Treino 3 épocas:")
    print(f"  gpt-4o     fine-tuning (~US$25/1M): ~US$ {tok*3/1e6*25:.2f}")
    print(f"  gpt-4o-mini fine-tuning (~US$3/1M): ~US$ {tok*3/1e6*3:.2f}")
    print("  (imagens dominam os tokens; valores aproximados — confirmar preço atual da OpenAI)")


if __name__ == "__main__":
    main()
