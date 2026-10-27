"""Piloto: few-shot de TEXTO + thinking, medindo custo real por caso.

Para cada caso alvo:
  1. recupera os K casos com QUEIXA mais parecida (cosine dos embeddings),
     excluindo o próprio caso — sem leak.
  2. monta o prompt com os LAUDOS desses K exemplos (texto, sem imagens) +
     a imagem SÓ do paciente novo.
  3. chama o modelo (Sonnet 5 ou Opus 4.8) com thinking adaptativo (effort high)
     e tool use p/ extrair ECGFindings.
  4. compara com o GT (mesmo cache do A/B) e registra tokens + custo.

Imagem do alvo: baseline jpeg cru (sem preprocess) — isola o efeito modelo+fewshot.

Uso:
  python pilot_fewshot.py --model sonnet --cases <uid1,uid2,...>
  python pilot_fewshot.py --model opus   --cases <mesmos uids>
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

os.environ["PREPROCESS_VISION"] = "0"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

import cv2
import numpy as np
import anthropic

from schemas import ECGFindings
from eval.comparator import compare

RESULTS = Path(__file__).resolve().parent / "results"
IMAGES = ROOT / "imagens"
EMB = RESULTS / "emb_index.json"
GT_CACHE = RESULTS / "gt_cache.json"

MODELS = {
    "sonnet": {"id": "claude-sonnet-5", "in": 3.0, "out": 15.0},   # preço PADRÃO
    "opus":   {"id": "claude-opus-4-8", "in": 5.0, "out": 25.0},
}

# system prompt enxuto (o da v2 é ótimo, mas repito o essencial + instrução de tool)
SYSTEM = """Você é cardiologista experiente em eletrocardiografia deste serviço.
Analise a foto do ECG do PACIENTE NOVO e os dados clínicos. Use os laudos-exemplo
de casos com queixa parecida apenas como referência de ESTILO e raciocínio — não
copie achados que você não enxergar no traçado do paciente novo.
Você DEVE chamar a tool extract_ecg_findings com os achados do paciente novo.
No campo resumo_clinico, escreva 2-4 frases no estilo de laudo da médica."""


def anthropic_key() -> str:
    txt = (ROOT / "system" / ".env").read_text()
    import re
    m = re.search(r"(?i)^\s*ANTHROPIC_API_KEY\s*=\s*(\S+)", txt, re.M)
    return m.group(1).strip().strip('"').strip("'")


def to_jpeg_b64(path: Path) -> str:
    arr = np.frombuffer(path.read_bytes(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.standard_b64encode(buf.tobytes()).decode()


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve(target, index, k):
    tv = target["vec"]
    scored = [(cosine(tv, c["vec"]), c) for c in index if c["uid"] != target["uid"]]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def build_content(target, neighbors):
    lines = ["Laudos-exemplo de casos com queixa parecida (referência de estilo):\n"]
    for i, nb in enumerate(neighbors, 1):
        q = nb["queixa"][:300].replace("\n", " ")
        lines.append(f"EXEMPLO {i} — queixa: {q}\n  Laudo da médica: {nb['laudo']}\n")
    lines.append("\n" + "=" * 40 + "\nPACIENTE NOVO (o caso a laudar):")
    q = target["queixa"][:1200].replace("\n", " ")
    lines.append(f"Dados clínicos: {q}\n\nFoto do ECG do paciente novo anexada. "
                 "Extraia os achados via tool extract_ecg_findings.")
    txt = "\n".join(lines)
    b64 = to_jpeg_b64(IMAGES / target["imagem"])
    return [
        {"type": "text", "text": txt},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--cases", required=True, help="uids separados por vírgula")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    mcfg = MODELS[args.model]
    index = json.loads(EMB.read_text())
    by_uid = {c["uid"]: c for c in index}
    gt_cache = json.loads(GT_CACHE.read_text())
    client = anthropic.Anthropic(api_key=anthropic_key())

    tool = {
        "name": "extract_ecg_findings",
        "description": "Estrutura os achados do ECG do paciente novo.",
        "input_schema": ECGFindings.model_json_schema(),
    }

    uids = [u.strip() for u in args.cases.split(",") if u.strip()]
    out = RESULTS / f"pilot_{args.model}.jsonl"
    out.write_text("")
    tot_in = tot_out = 0
    rows = []

    for uid in uids:
        if uid not in by_uid:
            print(f"[skip] uid desconhecido: {uid}"); continue
        target = by_uid[uid]
        nbs = retrieve(target, index, args.k)
        t0 = time.time()
        try:
            resp = client.messages.create(
                model=mcfg["id"],
                max_tokens=8000,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                tools=[tool],
                extra_body={"output_config": {"effort": "high"}},
                messages=[{"role": "user", "content": build_content(target, nbs)}],
            )
        except Exception as e:
            print(f"  {uid}: ERRO API {e}")
            rows.append({"uid": uid, "ok": False, "erro": str(e)})
            continue

        u = resp.usage
        tot_in += u.input_tokens
        tot_out += u.output_tokens
        findings = None
        for b in resp.content:
            if b.type == "tool_use" and b.name == "extract_ecg_findings":
                inp = b.input
                # às vezes o modelo aninha tudo sob 1 chave (ecg_findings/findings) — desembrulha
                if isinstance(inp, dict) and len(inp) == 1 and isinstance(next(iter(inp.values())), dict):
                    inp = next(iter(inp.values()))
                try:
                    findings = ECGFindings.model_validate(inp)
                except Exception as e:
                    print(f"  {uid}: tool JSON inválido {e}")
        rec = {"uid": uid, "idx": target["idx"], "stop": resp.stop_reason,
               "in": u.input_tokens, "out": u.output_tokens,
               "elapsed": round(time.time() - t0, 1)}
        if findings and uid in gt_cache:
            gt = ECGFindings.model_validate_json(gt_cache[uid])
            cmp = compare(findings, gt, case_id=target["idx"])
            rec.update(ok=True, counters={"agree": cmp.n_agree, "miss": cmp.n_miss,
                       "halluc": cmp.n_hallucination, "catch": cmp.n_catch},
                       resumo=findings.resumo_clinico)
            cc = rec["counters"]
            print(f"  {uid}: AGREE={cc['agree']} MISS={cc['miss']} HALLUC={cc['halluc']} "
                  f"| in={u.input_tokens} out={u.output_tokens} ({rec['elapsed']}s) [{resp.stop_reason}]")
        else:
            rec.update(ok=False, erro="sem findings ou sem GT")
            print(f"  {uid}: sem findings/GT | in={u.input_tokens} out={u.output_tokens}")
        rows.append(rec)
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len([r for r in rows if r.get("ok")])
    cost = tot_in / 1e6 * mcfg["in"] + tot_out / 1e6 * mcfg["out"]
    print(f"\n=== {args.model} ({mcfg['id']}) — {n}/{len(uids)} ok ===")
    print(f"tokens: in={tot_in} out={tot_out}")
    print(f"custo total: US$ {cost:.4f}  |  por caso: US$ {cost/max(len(uids),1):.4f} "
          f"(~R$ {cost/max(len(uids),1)*6:.2f})")
    ok = [r for r in rows if r.get("ok")]
    if ok:
        agg = {k: sum(r["counters"][k] for r in ok) for k in ("agree", "miss", "halluc", "catch")}
        print(f"TOTAIS: {agg}")


if __name__ == "__main__":
    main()
