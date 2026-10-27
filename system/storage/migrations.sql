-- Schema do banco. Roda 1x em db.init_db().
-- Separação crítica:
--   eval_cases     → 55 casos com GT (read-only após import; nunca usar p/ fewshot)
--   production_cases → casos novos do Telegram (fonte de retraining/fewshot)

CREATE TABLE IF NOT EXISTS eval_cases (
    id            INTEGER PRIMARY KEY,
    caso_texto    TEXT NOT NULL,
    laudo_gt      TEXT NOT NULL,
    conduta_gt    TEXT NOT NULL,
    data_original TEXT,
    imagem_path   TEXT NOT NULL,
    imported_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gt_findings_cache (
    case_id        INTEGER PRIMARY KEY REFERENCES eval_cases(id),
    findings_json  TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    parsed_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS production_cases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id   TEXT UNIQUE,
    caso_texto    TEXT NOT NULL,
    imagem_path   TEXT NOT NULL,
    chat_id       INTEGER,
    medico_user_id INTEGER,
    received_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    case_type        TEXT NOT NULL CHECK(case_type IN ('eval','production')),
    case_id          INTEGER NOT NULL,
    pipeline_version TEXT NOT NULL,
    response_json    TEXT NOT NULL,
    findings_json    TEXT,
    laudo            TEXT,
    conduta          TEXT,
    risco            TEXT,
    confianca_global REAL,
    tempo_ms         INTEGER,
    erro             TEXT,
    ran_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_case ON runs(case_type, case_id);

CREATE TABLE IF NOT EXISTS comparisons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    case_id         INTEGER NOT NULL,
    result_json     TEXT NOT NULL,
    n_agree         INTEGER NOT NULL DEFAULT 0,
    n_catch         INTEGER NOT NULL DEFAULT 0,
    n_miss          INTEGER NOT NULL DEFAULT 0,
    n_hallucination INTEGER NOT NULL DEFAULT 0,
    n_neutral       INTEGER NOT NULL DEFAULT 0,
    compared_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comparisons_case ON comparisons(case_id);

CREATE TABLE IF NOT EXISTS doctor_decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    production_case_id INTEGER NOT NULL REFERENCES production_cases(id),
    run_id             INTEGER NOT NULL REFERENCES runs(id),
    decision           TEXT NOT NULL CHECK(decision IN ('aprovado','corrigido','rejeitado')),
    laudo_final        TEXT,
    conduta_final      TEXT,
    correcao_texto     TEXT,
    medico_user_id     INTEGER,
    decided_at         TEXT NOT NULL,
    -- estilo: laudo/conduta clinicamente certos, só palavras diferentes
    -- clinico: achado ou conduta diferente do que o modelo produziu
    -- grave: correção que evitou prejuízo ao paciente (sinal de incidente)
    -- NULL: decisão anterior à migração ou aprovação sem correção
    correction_type    TEXT CHECK(correction_type IN ('estilo','clinico','grave'))
);

CREATE INDEX IF NOT EXISTS idx_decisions_prod ON doctor_decisions(production_case_id);

-- Reviews pendentes do bot Telegram (estado persistente entre reinícios)
CREATE TABLE IF NOT EXISTS pending_reviews (
    external_id        TEXT PRIMARY KEY,
    production_case_id INTEGER NOT NULL REFERENCES production_cases(id),
    run_id             INTEGER NOT NULL REFERENCES runs(id),
    review_msg_id      INTEGER,
    step               TEXT NOT NULL CHECK(step IN ('aguardando_decisao','aguardando_edicao')),
    laudo              TEXT NOT NULL,
    conduta            TEXT NOT NULL,
    risco              TEXT,
    patient_text       TEXT NOT NULL,
    image_path         TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    -- preenchido quando a médica escolhe 2/3/4 e o bot transita para
    -- aguardando_edicao; carregado para insert_decision junto da correção
    correction_type    TEXT CHECK(correction_type IN ('estilo','clinico','grave'))
);

CREATE INDEX IF NOT EXISTS idx_pending_msg ON pending_reviews(review_msg_id);

-- Inputs parciais do bot Telegram (foto OU texto chegou primeiro; aguardando o par).
-- Quando os dois chegam, processa e remove a linha.
CREATE TABLE IF NOT EXISTS pending_inputs (
    chat_id      INTEGER NOT NULL,
    user_id      INTEGER NOT NULL,
    image_path   TEXT,
    caption_text TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS catch_reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id INTEGER NOT NULL REFERENCES comparisons(id),
    case_id       INTEGER NOT NULL,
    campo         TEXT NOT NULL,
    sistema       TEXT NOT NULL,
    gt            TEXT NOT NULL,
    veredito      TEXT CHECK(veredito IN ('verdadeiro','falso','irrelevante')),
    observacao    TEXT,
    reviewed_at   TEXT
);
