"""Few-shot dinâmico para captar o estilo de escrita da médica na conduta.

Pool de exemplos = `eval_cases` (laudo/conduta — TODOS escritos/validados pela
mesma médica) + decisões de produção `corrigido`. Tudo é a voz dela; não há
distinção de "voz" entre v1 e produção.

RECUPERAÇÃO POR QUADRO CLÍNICO (não só queixa): a conduta depende de
achados do ECG + risco + queixa juntos — a mesma queixa ("dor torácica") pede
condutas opostas se o ECG é normal vs. tem supra de ST. Por isso o score pesa:
  - padrões do ECG (classify) — dominante; é o que dita a conduta
  - risco (BAIXO..EMERGENCIA)  — mesma urgência → mesma família de conduta
  - queixa (similaridade lexical) — contexto/desempate

Sinais estruturados: v1 vem de `gt_findings_cache`; produção de `runs`. Onde o
risco não está armazenado (v1), deriva-se dos padrões. Lê do banco com cache TTL
curto → correções novas entram no pool ao vivo, sem reiniciar o bot.

Fallback: qualquer falha de parse/DB degrada para menos sinais, nunca derruba."""

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from schemas import PatientData
from schemas.findings import ECGFindings
from rules.pattern_classifier import classify
from rules.risk_engine import compute_risk

_TOK = re.compile(r"[a-zà-ÿ]+")
_TTL = 120.0  # s — correções novas entram no pool em até ~2min

# Pesos do score de recuperação (quadro clínico domina; queixa é contexto).
W_PAT, W_RISK, W_Q = 0.55, 0.20, 0.25

_CACHE: list[dict] = []
_CACHE_TS: float = 0.0


def _toks(s: str) -> set[str]:
    return {t for t in _TOK.findall((s or "").lower()) if len(t) > 3}


def _jac(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _padroes_risco(findings_json, risco_armazenado: str | None
                   ) -> tuple[set[str], str, ECGFindings | None]:
    """(padrões, risco, findings) a partir do findings_json. Usa o risco armazenado
    se houver (produção); senão deriva dos padrões (v1). Falha de parse → vazios."""
    if not findings_json:
        return set(), (risco_armazenado or ""), None
    try:
        raw = findings_json if isinstance(findings_json, str) else json.dumps(findings_json)
        f = ECGFindings.model_validate_json(raw)
        pad = set(classify(f))
        if risco_armazenado:
            return pad, risco_armazenado, f
        risco, _, _ = compute_risk(list(pad), PatientData())  # risco baseado só nos padrões
        return pad, risco, f
    except Exception:
        return set(), (risco_armazenado or ""), None


def _carregar() -> list[dict]:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    ex: list[dict] = []

    for r in c.execute(
        """SELECT e.caso_texto, e.laudo_gt, e.conduta_gt, g.findings_json
             FROM eval_cases e
             LEFT JOIN gt_findings_cache g ON g.case_id = e.id
            WHERE TRIM(COALESCE(e.laudo_gt,''))<>'' AND TRIM(COALESCE(e.conduta_gt,''))<>''"""
    ):
        pad, risco, f = _padroes_risco(r["findings_json"], None)
        ex.append({"caso": r["caso_texto"], "laudo": r["laudo_gt"], "conduta": r["conduta_gt"],
                   "_q": _toks(r["caso_texto"]), "_pat": pad, "_risco": risco,
                   "_findings": f, "_ctype": None, "_source": "v1"})

    for r in c.execute(
        """SELECT p.caso_texto, d.laudo_final, d.conduta_final,
                  r.findings_json, r.risco, d.correction_type
             FROM doctor_decisions d
             JOIN production_cases p ON p.id = d.production_case_id
             LEFT JOIN runs r ON r.id = d.run_id
            WHERE d.decision='corrigido'
              AND d.id IN (SELECT MAX(id) FROM doctor_decisions
                           WHERE decision='corrigido' GROUP BY production_case_id)
              AND TRIM(COALESCE(d.conduta_final,''))<>''"""
    ):
        pad, risco, f = _padroes_risco(r["findings_json"], r["risco"])
        ex.append({"caso": r["caso_texto"], "laudo": r["laudo_final"], "conduta": r["conduta_final"],
                   "_q": _toks(r["caso_texto"]), "_pat": pad, "_risco": risco,
                   "_findings": f, "_ctype": r["correction_type"], "_source": "prod"})

    c.close()
    return ex


def _pool() -> list[dict]:
    global _CACHE, _CACHE_TS
    now = time.time()
    if not _CACHE or now - _CACHE_TS > _TTL:
        try:
            _CACHE = _carregar()
            _CACHE_TS = now
        except Exception:
            pass  # erro de DB → mantém o cache anterior; nunca derruba o pipeline
    return _CACHE


def _score(e: dict, tq: set, pq: set, rq: str) -> float:
    return (W_PAT * _jac(pq, e["_pat"])
            + W_Q * _jac(tq, e["_q"])
            + (W_RISK if (rq and rq == e["_risco"]) else 0.0))


def retrieve(queixa: str, patterns: list[str] | None = None,
             risco: str | None = None, k: int = 3) -> list[dict]:
    """Top-k exemplos pelo QUADRO CLÍNICO (padrões + risco + queixa). Ordenados do
    MENOS para o MAIS parecido — o mais parecido fica por último, posição de maior
    influência no few-shot. Retorna [] se nada casar."""
    tq, pq, rq = _toks(queixa), set(patterns or []), (risco or "")
    scored = [(s, e) for e in _pool() if (s := _score(e, tq, pq, rq)) > 0.0]
    scored.sort(key=lambda x: x[0])
    top = scored[-k:] if len(scored) > k else scored
    return [e for _, e in top]


def retrieve_laudo(patterns: list[str] | None = None, risco: str | None = None,
                   k: int = 3) -> list[dict]:
    """Top-k para fewshot de LAUDO. Filtra exemplos que têm findings parseáveis
    (precisamos para regenerar o "laudo bruto" do exemplo via template). O
    laudo é função do ECG, não do caso clínico — por isso ignora a queixa
    (e re-normaliza os pesos só entre padrão+risco)."""
    pq, rq = set(patterns or []), (risco or "")
    # mesmo critério do retrieve(), mas sem o componente de queixa: laudo
    # descreve o ECG, e a queixa só dilui o sinal aqui.
    scored = []
    denom = W_PAT + W_RISK
    for e in _pool():
        if e["_findings"] is None or not e["laudo"]:
            continue
        s = W_PAT * _jac(pq, e["_pat"])
        if rq and rq == e["_risco"]:
            s += W_RISK
        s = s / denom  # normaliza pra [0,1]
        if s <= 0.0:
            continue
        scored.append((s, e))
    scored.sort(key=lambda x: x[0])
    top = scored[-k:] if len(scored) > k else scored
    return [e for _, e in top]
