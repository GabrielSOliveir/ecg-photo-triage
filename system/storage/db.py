import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH

_MIGRATIONS = Path(__file__).parent / "migrations.sql"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    sql = _MIGRATIONS.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(sql)
        _migrate_add_correction_type(conn)


def _migrate_add_correction_type(conn: sqlite3.Connection) -> None:
    """Adiciona correction_type em doctor_decisions e pending_reviews para bancos
    criados antes da feature dos botões 2/3/4. SQLite não tem ADD COLUMN IF NOT
    EXISTS, então checa PRAGMA table_info antes."""
    for table in ("doctor_decisions", "pending_reviews"):
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "correction_type" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN correction_type TEXT")


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# ── eval_cases ────────────────────────────────────────────────────────────────


def import_eval_case(
    case_id: int,
    caso_texto: str,
    laudo_gt: str,
    conduta_gt: str,
    data_original: Optional[str],
    imagem_path: str,
) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO eval_cases
               (id, caso_texto, laudo_gt, conduta_gt, data_original, imagem_path, imported_at)
               VALUES (?,?,?,?,?,?,?)""",
            (case_id, caso_texto, laudo_gt, conduta_gt, data_original, imagem_path, _now()),
        )


def list_eval_cases() -> list[sqlite3.Row]:
    with connect() as conn:
        return list(conn.execute("SELECT * FROM eval_cases ORDER BY id"))


def get_eval_case(case_id: int) -> Optional[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM eval_cases WHERE id=?", (case_id,)).fetchone()


# ── gt_findings_cache ─────────────────────────────────────────────────────────


def save_gt_findings(case_id: int, findings_json: str, parser_version: str) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO gt_findings_cache
               (case_id, findings_json, parser_version, parsed_at)
               VALUES (?,?,?,?)""",
            (case_id, findings_json, parser_version, _now()),
        )


def get_gt_findings(case_id: int) -> Optional[str]:
    with connect() as conn:
        row = conn.execute(
            "SELECT findings_json FROM gt_findings_cache WHERE case_id=?",
            (case_id,),
        ).fetchone()
        return row["findings_json"] if row else None


# ── production_cases ──────────────────────────────────────────────────────────


def insert_production_case(
    external_id: str,
    caso_texto: str,
    imagem_path: str,
    chat_id: Optional[int],
    medico_user_id: Optional[int],
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO production_cases
               (external_id, caso_texto, imagem_path, chat_id, medico_user_id, received_at)
               VALUES (?,?,?,?,?,?)""",
            (external_id, caso_texto, imagem_path, chat_id, medico_user_id, _now()),
        )
        return cur.lastrowid


def get_production_case_by_external(external_id: str) -> Optional[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM production_cases WHERE external_id=?",
            (external_id,),
        ).fetchone()


# ── runs ──────────────────────────────────────────────────────────────────────


def insert_run(
    case_type: str,
    case_id: int,
    pipeline_version: str,
    response_dict: dict,
) -> int:
    findings_json = (
        json.dumps(response_dict["findings"], ensure_ascii=False)
        if response_dict.get("findings")
        else None
    )
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO runs
               (case_type, case_id, pipeline_version, response_json,
                findings_json, laudo, conduta, risco, confianca_global,
                tempo_ms, erro, ran_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                case_type,
                case_id,
                pipeline_version,
                json.dumps(response_dict, ensure_ascii=False),
                findings_json,
                response_dict.get("laudo"),
                response_dict.get("conduta"),
                response_dict.get("risco"),
                response_dict.get("confianca_global"),
                response_dict.get("tempo_ms"),
                response_dict.get("erro"),
                _now(),
            ),
        )
        return cur.lastrowid


def latest_run_for(case_type: str, case_id: int) -> Optional[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            """SELECT * FROM runs
               WHERE case_type=? AND case_id=?
               ORDER BY id DESC LIMIT 1""",
            (case_type, case_id),
        ).fetchone()


# ── comparisons ───────────────────────────────────────────────────────────────


def insert_comparison(
    run_id: int,
    case_id: int,
    result_dict: dict,
    counters: dict,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO comparisons
               (run_id, case_id, result_json,
                n_agree, n_catch, n_miss, n_hallucination, n_neutral, compared_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                case_id,
                json.dumps(result_dict, ensure_ascii=False),
                counters.get("n_agree", 0),
                counters.get("n_catch", 0),
                counters.get("n_miss", 0),
                counters.get("n_hallucination", 0),
                counters.get("n_neutral", 0),
                _now(),
            ),
        )
        return cur.lastrowid


def list_comparisons() -> list[sqlite3.Row]:
    with connect() as conn:
        return list(conn.execute("SELECT * FROM comparisons ORDER BY case_id"))


# ── doctor_decisions ──────────────────────────────────────────────────────────


def insert_decision(
    production_case_id: int,
    run_id: int,
    decision: str,
    laudo_final: Optional[str],
    conduta_final: Optional[str],
    correcao_texto: Optional[str],
    medico_user_id: Optional[int],
    correction_type: Optional[str] = None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO doctor_decisions
               (production_case_id, run_id, decision, laudo_final, conduta_final,
                correcao_texto, medico_user_id, decided_at, correction_type)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                production_case_id,
                run_id,
                decision,
                laudo_final,
                conduta_final,
                correcao_texto,
                medico_user_id,
                _now(),
                correction_type,
            ),
        )
        return cur.lastrowid


# ── pending_reviews (bot Telegram, estado persistente) ───────────────────────


def upsert_pending_review(
    external_id: str,
    production_case_id: int,
    run_id: int,
    review_msg_id: Optional[int],
    step: str,
    laudo: str,
    conduta: str,
    risco: str,
    patient_text: str,
    image_path: str,
) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO pending_reviews
               (external_id, production_case_id, run_id, review_msg_id, step,
                laudo, conduta, risco, patient_text, image_path,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(external_id) DO UPDATE SET
                   review_msg_id=excluded.review_msg_id,
                   step=excluded.step,
                   updated_at=excluded.updated_at""",
            (external_id, production_case_id, run_id, review_msg_id, step,
             laudo, conduta, risco, patient_text, image_path,
             _now(), _now()),
        )


def get_pending_by_msg(review_msg_id: int) -> Optional[sqlite3.Row]:
    """Retorna pending + `medico_user_id` da production_case (owner), para o
    bot checar se quem está respondendo é a mesma médica que enviou a foto."""
    with connect() as conn:
        return conn.execute(
            """SELECT pr.*, pc.medico_user_id
               FROM pending_reviews pr
               JOIN production_cases pc ON pc.id = pr.production_case_id
               WHERE pr.review_msg_id=?""",
            (review_msg_id,),
        ).fetchone()


def get_pending_by_external(external_id: str) -> Optional[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            """SELECT pr.*, pc.medico_user_id
               FROM pending_reviews pr
               JOIN production_cases pc ON pc.id = pr.production_case_id
               WHERE pr.external_id=?""",
            (external_id,),
        ).fetchone()


def list_pending_for_user(user_id: int) -> list[sqlite3.Row]:
    """Pendentes cuja production_case foi criada por este médico.

    Permite o fluxo rápido (responder '1' ou '2' sem reply) quando a médica
    tem apenas um caso aberto, sem confundir com casos de outras médicas.
    """
    with connect() as conn:
        return list(conn.execute(
            """SELECT pr.* FROM pending_reviews pr
               JOIN production_cases pc ON pc.id = pr.production_case_id
               WHERE pc.medico_user_id = ?
               ORDER BY pr.updated_at DESC""",
            (user_id,),
        ))


def set_pending_step(external_id: str, step: str,
                     correction_type: Optional[str] = None) -> None:
    """Atualiza o step e (opcionalmente) registra que tipo de correção a médica
    selecionou ao apertar 2/3/4 — esse tipo é lido depois quando a edição chegar.
    Passar `correction_type=None` deixa a coluna como está (não sobrescreve)."""
    with connect() as conn:
        if correction_type is None:
            conn.execute(
                "UPDATE pending_reviews SET step=?, updated_at=? WHERE external_id=?",
                (step, _now(), external_id),
            )
        else:
            conn.execute(
                """UPDATE pending_reviews
                      SET step=?, correction_type=?, updated_at=?
                    WHERE external_id=?""",
                (step, correction_type, _now(), external_id),
            )


def delete_pending(external_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM pending_reviews WHERE external_id=?", (external_id,))


# ── pending_inputs (bot Telegram: foto OU texto chegou; aguardando o outro) ──


def upsert_pending_input(
    chat_id: int,
    user_id: int,
    image_path: Optional[str],
    caption_text: Optional[str],
) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO pending_inputs
               (chat_id, user_id, image_path, caption_text, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(chat_id, user_id) DO UPDATE SET
                   image_path=excluded.image_path,
                   caption_text=excluded.caption_text,
                   created_at=excluded.created_at""",
            (chat_id, user_id, image_path, caption_text, _now()),
        )


def get_pending_input(chat_id: int, user_id: int) -> Optional[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM pending_inputs WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        ).fetchone()


def delete_pending_input(chat_id: int, user_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM pending_inputs WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        )


# ── catch_reviews ─────────────────────────────────────────────────────────────


def insert_catch_review(
    comparison_id: int,
    case_id: int,
    campo: str,
    sistema: str,
    gt: str,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO catch_reviews
               (comparison_id, case_id, campo, sistema, gt)
               VALUES (?,?,?,?,?)""",
            (comparison_id, case_id, campo, sistema, gt),
        )
        return cur.lastrowid


def set_catch_verdict(
    review_id: int,
    veredito: str,
    observacao: Optional[str] = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE catch_reviews
               SET veredito=?, observacao=?, reviewed_at=?
               WHERE id=?""",
            (veredito, observacao, _now(), review_id),
        )


def list_catch_reviews(case_id: Optional[int] = None) -> list[sqlite3.Row]:
    with connect() as conn:
        if case_id is not None:
            return list(conn.execute(
                "SELECT * FROM catch_reviews WHERE case_id=? ORDER BY id",
                (case_id,),
            ))
        return list(conn.execute("SELECT * FROM catch_reviews ORDER BY case_id, id"))
