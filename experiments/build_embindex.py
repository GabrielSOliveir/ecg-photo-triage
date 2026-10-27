"""Índice de embeddings das QUEIXAS (caso_texto) p/ retrieval de few-shot.

Usa OpenAI text-embedding-3-small (barato, ~1 centavo pro dataset inteiro).
Lê a chave do .env raiz (variável 'openai_API_KEY', minúscula — nome fora do
padrão, então extraímos manualmente).

Salva results/emb_index.json: lista de {uid, idx, id, origem, queixa, laudo, vec}.
"""

import json
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "dataset" / "casos_unificados.xlsx"
IMAGES = ROOT / "imagens"
OUT = Path(__file__).resolve().parent / "results" / "emb_index.json"
MODEL = "text-embedding-3-small"


def openai_key() -> str:
    txt = (ROOT / ".env").read_text()
    m = re.search(r"(?i)^\s*openai_API_KEY\s*=\s*(\S+)", txt, re.M)
    if not m:
        raise RuntimeError("openai_API_KEY não encontrada no .env raiz")
    return m.group(1).strip().strip('"').strip("'")


def load_cases():
    ws = openpyxl.load_workbook(XLSX, read_only=True).active
    header = None
    cases = []
    for r in ws.iter_rows(values_only=True):
        if header is None:
            header = list(r)
            continue
        row = dict(zip(header, r))
        img, laudo = row.get("imagem"), row.get("laudo")
        if not img or not laudo or not (IMAGES / str(img)).exists():
            continue
        origem = str(row.get("origem") or "?")
        cases.append({
            "uid": f"{origem}:{row['id']}",
            "idx": len(cases),
            "id": str(row["id"]),
            "origem": origem,
            "queixa": str(row.get("caso_texto") or ""),
            "laudo": str(laudo),
            "imagem": str(img),
        })
    return cases


def main():
    from openai import OpenAI
    client = OpenAI(api_key=openai_key())
    cases = load_cases()
    texts = [c["queixa"][:4000] or "(sem queixa)" for c in cases]
    print(f"Embedando {len(texts)} queixas com {MODEL}...")
    # a API aceita lista; mandamos em lotes de 128 por segurança
    vecs = []
    for i in range(0, len(texts), 128):
        chunk = texts[i:i + 128]
        resp = client.embeddings.create(model=MODEL, input=chunk)
        vecs.extend([d.embedding for d in resp.data])
        print(f"  {min(i+128, len(texts))}/{len(texts)}")
    for c, v in zip(cases, vecs):
        c["vec"] = v
    OUT.write_text(json.dumps(cases, ensure_ascii=False))
    tot_tokens = sum(len(t.split()) for t in texts)
    print(f"OK → {OUT.name} ({len(cases)} casos, dim={len(vecs[0])})")
    print(f"~{tot_tokens} palavras embedadas (custo << 1 centavo)")


if __name__ == "__main__":
    main()
