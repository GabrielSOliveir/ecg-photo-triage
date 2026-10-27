"""Juiz de aceitação: para cada caso, decide se a médica aceitaria o laudo
e a conduta gerados pelo sistema, dado o GT da cardiologista como referência.

Usa Haiku via tool use. Saída em SQLite (tabela acceptance_judgments) e CSV
para a médica calibrar uma amostra."""

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from pydantic import BaseModel, Field

from config import ANTHROPIC_API_KEY, GT_PARSE_MODEL
from storage import db

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

JUDGE_VERSION = "judge-v2-style"
Veredito = Literal["ACEITARIA", "ACEITARIA_COM_EDICAO", "REJEITARIA"]
# Eixo separado de estilo: o veredito mede só clínica (achados/conduta correta).
# Estilo mede aderência à voz da médica — duas escritas clinicamente equivalentes
# podem ter aderência diferente (frases longas vs telegráficas, siglas expandidas
# ou não, ordem de achados, expressões dela como "secundárias a", "padrão de strain").
EstiloAderencia = Literal["ALTA", "MEDIA", "BAIXA"]


class JudgeOutput(BaseModel):
    veredito_laudo: Veredito
    motivo_laudo: str = Field(..., description="≤120 caracteres, justificando")
    veredito_conduta: Veredito
    motivo_conduta: str = Field(..., description="≤120 caracteres, justificando")
    style_aderence_laudo: EstiloAderencia = Field(
        ..., description="Aderência ao estilo da médica (voz/estrutura), independe da clínica"
    )
    style_aderence_conduta: EstiloAderencia = Field(
        ..., description="Aderência ao estilo da médica na conduta (mesmo critério)"
    )


JUDGE_SYSTEM = """Você é um médico sênior auditando se um colega plantonista, com 2 min por ECG,
aceitaria a sugestão do sistema sem reescrever.

Você responde DUAS dimensões INDEPENDENTES para LAUDO e para CONDUTA:

═══ DIMENSÃO 1 — VEREDITO CLÍNICO (essência) ═══

Ignore estilo aqui. Foque APENAS em conteúdo clínico:
  ACEITARIA            — clinicamente equivalente ao GT, plantonista publicaria.
  ACEITARIA_COM_EDICAO — semelhante, ajuste pequeno de termo/dose/ordem (<30s).
  REJEITARIA           — clinicamente diferente (achado errado, conduta inadequada).

Se sistema for MAIS DETALHADO mas correto, é ACEITARIA.
Se sistema mencionar achado/conduta NÃO PRESENTE NO GT: clinicamente plausível
→ ACEITARIA_COM_EDICAO; implausível/errado → REJEITARIA.

═══ DIMENSÃO 2 — ADERÊNCIA AO ESTILO DA MÉDICA (voz) ═══

Compare a FORMA com a do GT (que é o que essa médica/cardiologista escreveria):
  ALTA   — frase/estrutura/expressões soam como se ela tivesse escrito.
  MEDIA  — clinicamente certo mas tom diferente (mais longo, mais hedge,
           termo técnico no lugar do coloquial dela, ordem trocada).
  BAIXA  — soa de outro autor — frases longas didáticas onde ela seria
           telegráfica, ou termos que ela nunca usa.

MARCADORES do estilo dela observados no GT:
- LAUDO: telegráfico, achados em sequência ("Ritmo sinusal, FC 92 bpm. Eixo
  preservado. Sem sinais de isquemia/sobrecarga."), uso de "secundárias a X",
  "compatível com", "padrão de strain", "alteração inespecífica de
  repolarização".
- CONDUTA: diferenciais nomeados em lista ("IC? DPOC? Pneumonia? TEP?"),
  expansão de siglas em parênteses ("DPOC (doenca pulmonar obstrutiva
  cronica)"), começa com "Sugiro/Considerar/Avaliar", múltiplas linhas curtas.

Avalie estilo INDEPENDENTE da clínica: um laudo clinicamente perfeito mas
escrito como livro-texto pode ter aderência BAIXA. Um laudo telegráfico
clinicamente errado tem clínica REJEITARIA mas estilo ALTA — são eixos
ortogonais.

Responda via tool judge."""


JUDGE_USER = """CONTEXTO DO PACIENTE:
{caso}

LAUDO DO CARDIOLOGISTA (referência):
{laudo_gt}

LAUDO DO SISTEMA:
{laudo_sis}

CONDUTA DO CARDIOLOGISTA (referência):
{conduta_gt}

CONDUTA DO SISTEMA:
{conduta_sis}

Julgue via tool."""


def _ensure_table():
    with db.connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS acceptance_judgments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id         INTEGER NOT NULL,
                run_id          INTEGER NOT NULL,
                judge_version   TEXT NOT NULL,
                veredito_laudo  TEXT NOT NULL,
                motivo_laudo    TEXT,
                veredito_conduta TEXT NOT NULL,
                motivo_conduta  TEXT,
                judged_at       TEXT NOT NULL,
                medico_veredito_laudo TEXT,
                medico_veredito_conduta TEXT,
                medico_obs      TEXT
            )
        """)
        # judge-v2: adiciona aderência de estilo. Idempotente.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(acceptance_judgments)")}
        for col in ("style_aderence_laudo", "style_aderence_conduta"):
            if col not in cols:
                conn.execute(f"ALTER TABLE acceptance_judgments ADD COLUMN {col} TEXT")


def judge_case(case_id: int, run_row: sqlite3.Row, case_row: sqlite3.Row) -> JudgeOutput:
    tool = {
        "name": "judge",
        "description": "Avalia se médica aceitaria laudo e conduta.",
        "input_schema": JudgeOutput.model_json_schema(),
    }
    user = JUDGE_USER.format(
        caso=case_row["caso_texto"],
        laudo_gt=case_row["laudo_gt"],
        laudo_sis=run_row["laudo"] or "(vazio)",
        conduta_gt=case_row["conduta_gt"],
        conduta_sis=run_row["conduta"] or "(vazio)",
    )
    response = _client.messages.create(
        model=GT_PARSE_MODEL,
        max_tokens=1024,
        temperature=0,
        system=JUDGE_SYSTEM,
        tools=[tool],
        tool_choice={"type": "tool", "name": "judge"},
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return JudgeOutput.model_validate(block.input)
    raise RuntimeError("Modelo não invocou a tool")


def save_judgment(case_id: int, run_id: int, j: JudgeOutput) -> int:
    from datetime import datetime
    with db.connect() as conn:
        cur = conn.execute(
            """INSERT INTO acceptance_judgments
               (case_id, run_id, judge_version, veredito_laudo, motivo_laudo,
                veredito_conduta, motivo_conduta, judged_at,
                style_aderence_laudo, style_aderence_conduta)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (case_id, run_id, JUDGE_VERSION,
             j.veredito_laudo, j.motivo_laudo,
             j.veredito_conduta, j.motivo_conduta,
             datetime.utcnow().isoformat(timespec="seconds"),
             j.style_aderence_laudo, j.style_aderence_conduta),
        )
        return cur.lastrowid


def run_all(force: bool = False):
    db.init_db()
    _ensure_table()

    cases = db.list_eval_cases()
    print(f"Julgando {len(cases)} casos...")
    ok = err = skipped = 0

    for c in cases:
        cid = c["id"]
        with db.connect() as conn:
            already = conn.execute(
                "SELECT 1 FROM acceptance_judgments WHERE case_id=? AND judge_version=?",
                (cid, JUDGE_VERSION),
            ).fetchone()
        if already and not force:
            skipped += 1
            continue

        run = db.latest_run_for("eval", cid)
        if not run or not run["laudo"] or not run["conduta"]:
            print(f"  [skip] caso {cid}: sem run com laudo/conduta")
            err += 1
            continue

        try:
            t0 = time.time()
            j = judge_case(cid, run, c)
            save_judgment(cid, run["id"], j)
            print(f"  caso {cid}: L={j.veredito_laudo[:8]}/{j.style_aderence_laudo}"
                  f" | C={j.veredito_conduta[:8]}/{j.style_aderence_conduta}"
                  f"  ({time.time()-t0:.1f}s)")
            ok += 1
        except Exception as e:
            print(f"  [erro] caso {cid}: {e}")
            err += 1

    print(f"\nOK: {ok} | Skip: {skipped} | Erro: {err}")


def export_csv(path: str):
    with db.connect() as conn:
        rows = list(conn.execute("""
            SELECT j.id, j.case_id, j.veredito_laudo, j.motivo_laudo,
                   j.veredito_conduta, j.motivo_conduta,
                   c.caso_texto, c.laudo_gt, c.conduta_gt,
                   r.laudo AS laudo_sistema, r.conduta AS conduta_sistema
            FROM acceptance_judgments j
            JOIN eval_cases c ON c.id = j.case_id
            JOIN runs r ON r.id = j.run_id
            ORDER BY j.case_id
        """))
    cols = ["judgment_id", "case_id",
            "veredito_laudo", "motivo_laudo",
            "veredito_conduta", "motivo_conduta",
            "caso", "laudo_gt", "laudo_sistema",
            "conduta_gt", "conduta_sistema",
            "medico_veredito_laudo", "medico_veredito_conduta", "medico_obs"]
    with open(path, "w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "judgment_id": r["id"],
                "case_id": r["case_id"],
                "veredito_laudo": r["veredito_laudo"],
                "motivo_laudo": r["motivo_laudo"],
                "veredito_conduta": r["veredito_conduta"],
                "motivo_conduta": r["motivo_conduta"],
                "caso": r["caso_texto"],
                "laudo_gt": r["laudo_gt"],
                "laudo_sistema": r["laudo_sistema"],
                "conduta_gt": r["conduta_gt"],
                "conduta_sistema": r["conduta_sistema"],
                "medico_veredito_laudo": "",
                "medico_veredito_conduta": "",
                "medico_obs": "",
            })
    print(f"✓ {len(rows)} linhas em {path}")


def report():
    with db.connect() as conn:
        rows = list(conn.execute("""
            SELECT case_id, veredito_laudo, veredito_conduta,
                   style_aderence_laudo, style_aderence_conduta
            FROM acceptance_judgments
            WHERE judge_version=?
        """, (JUDGE_VERSION,)))

    n = len(rows)
    if n == 0:
        print("Sem judgments. Rode: python -m eval.acceptance --run")
        return

    perfeito = meio_laudo = meio_conduta = zero = 0
    laudo_ok = conduta_ok = 0
    for r in rows:
        l_aceita = r["veredito_laudo"] in ("ACEITARIA", "ACEITARIA_COM_EDICAO")
        c_aceita = r["veredito_conduta"] in ("ACEITARIA", "ACEITARIA_COM_EDICAO")
        if l_aceita and c_aceita:
            perfeito += 1
        elif l_aceita:
            meio_laudo += 1
        elif c_aceita:
            meio_conduta += 1
        else:
            zero += 1
        if l_aceita: laudo_ok += 1
        if c_aceita: conduta_ok += 1

    perfeito_strict = sum(1 for r in rows
                          if r["veredito_laudo"] == "ACEITARIA"
                          and r["veredito_conduta"] == "ACEITARIA")

    print("=" * 60)
    print(f"ACEITAÇÃO — {n} casos (juiz LLM)")
    print("=" * 60)
    print()
    print("CENÁRIO IDEAL (médica aceita sem editar):")
    print(f"  → PERFEITO ESTRITO (laudo OK + conduta OK, sem edição): {perfeito_strict}/{n}  ({100*perfeito_strict/n:.0f}%)")
    print()
    print("CENÁRIO REAL (incluindo edições <30s):")
    print(f"  → PERFEITO: {perfeito}/{n}  ({100*perfeito/n:.0f}%)  — aceita ambos")
    print(f"  → MEIO LAUDO: {meio_laudo}/{n}  ({100*meio_laudo/n:.0f}%)  — aceita só laudo")
    print(f"  → MEIO CONDUTA: {meio_conduta}/{n}  ({100*meio_conduta/n:.0f}%)  — aceita só conduta")
    print(f"  → ZERO: {zero}/{n}  ({100*zero/n:.0f}%)  — refaz tudo")
    print()
    print(f"PROVEITO em pelo menos 1 dos 2: {perfeito + meio_laudo + meio_conduta}/{n}  "
          f"({100*(perfeito+meio_laudo+meio_conduta)/n:.0f}%)")
    print()
    print(f"Taxa por componente:")
    print(f"  laudo aceitável:   {laudo_ok}/{n}  ({100*laudo_ok/n:.0f}%)")
    print(f"  conduta aceitável: {conduta_ok}/{n}  ({100*conduta_ok/n:.0f}%)")
    print()
    print("Detalhamento veredito laudo:")
    for v in ("ACEITARIA", "ACEITARIA_COM_EDICAO", "REJEITARIA"):
        c = sum(1 for r in rows if r["veredito_laudo"] == v)
        print(f"  {v:25s} {c:3d}  ({100*c/n:.0f}%)")
    print()
    print("Detalhamento veredito conduta:")
    for v in ("ACEITARIA", "ACEITARIA_COM_EDICAO", "REJEITARIA"):
        c = sum(1 for r in rows if r["veredito_conduta"] == v)
        print(f"  {v:25s} {c:3d}  ({100*c/n:.0f}%)")

    print()
    print("─" * 60)
    print("ADERÊNCIA AO ESTILO DA MÉDICA (eixo separado)")
    print("─" * 60)
    com_estilo = [r for r in rows if r["style_aderence_laudo"]]  # judge-v2 ou +
    if not com_estilo:
        print("  (sem julgamentos de estilo — re-rode com --force para judge-v2)")
        return
    n2 = len(com_estilo)
    for campo in ("style_aderence_laudo", "style_aderence_conduta"):
        rotulo = "Laudo  " if campo.endswith("laudo") else "Conduta"
        print(f"  {rotulo}:")
        for v in ("ALTA", "MEDIA", "BAIXA"):
            c = sum(1 for r in com_estilo if r[campo] == v)
            print(f"     {v:5s}: {c:3d}/{n2}  ({100*c/n2:.0f}%)")
    alta_alta = sum(1 for r in com_estilo
                    if r["style_aderence_laudo"] == "ALTA"
                    and r["style_aderence_conduta"] == "ALTA")
    print(f"  → Estilo ALTA em ambos: {alta_alta}/{n2}  ({100*alta_alta/n2:.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="roda o juiz nos 54 casos")
    parser.add_argument("--force", action="store_true", help="re-julga mesmo já julgados")
    parser.add_argument("--report", action="store_true", help="mostra agregado")
    parser.add_argument("--export", metavar="PATH", help="exporta CSV para revisão médica")
    args = parser.parse_args()

    if args.run:
        run_all(force=args.force)
    if args.report:
        report()
    if args.export:
        export_csv(args.export)
    if not (args.run or args.report or args.export):
        parser.print_help()


if __name__ == "__main__":
    main()
