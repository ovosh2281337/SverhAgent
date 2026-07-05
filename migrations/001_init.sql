-- Knowledge-collection prototype schema.
-- Principle: messages is raw truth, never mutated. Everything else is a derived
-- layer that a new parser can regenerate at any time.

-- pgvector powers dedup-on-insert and the search_knowledge tool. The docker
-- image is pgvector/pgvector; a local Postgres must have the extension available.
-- Embedding dim is 640 (local harrier-oss-v1-270m); change vector(640) below if
-- you swap EMBED_MODEL for one with a different dimension.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sessions (
    id           BIGSERIAL PRIMARY KEY,
    expert_name  TEXT        NOT NULL,
    topic        TEXT        NOT NULL,
    -- status machine: active | finished | extracting | extracted.
    -- finished->extracting is an atomic CAS (see db.claim_for_extraction) that
    -- makes double extraction impossible.
    status       TEXT        NOT NULL DEFAULT 'active',
    -- which dialogue system prompt produced this session; lets us A/B iterate.
    prompt_version TEXT      NOT NULL DEFAULT 'v1',
    -- running token spend, accumulated by the wrapper for the session budget.
    tokens_used  BIGINT      NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);

-- Source of truth. Never modified after insert.
-- role: system | user | assistant. content is the visible text; tool_calls holds
-- the full Anthropic content-block array for assistant turns (audit / re-parse).
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT      NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL DEFAULT '',
    tool_calls  JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

-- Interview plan and coverage. Advisory map-hint, not source of truth.
-- status: open | covered
CREATE TABLE IF NOT EXISTS plan_items (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT      NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    subtopic    TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'open',
    ord         INT         NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_plan_session ON plan_items(session_id, ord);

-- Derived layer, regenerable. type: fact | qa_pair | term.
-- origin (silent-corruption guard, structural not prompt-only):
--   expert_claim         — stated by the expert; only these count as facts.
--   confirmed_hypothesis — an agent hypothesis the expert explicitly confirmed;
--                          weak weight, flagged as such in the summary.
-- Implicit non-disputes are not extracted as facts at all (become open questions).
-- quote + source_message_id anchor every item so any claim is verifiable in secs.
-- Dedup-on-insert: embedding drives nearest-canonical lookup; a near-duplicate
-- gets duplicate_of set (points at the canonical row) and bumps that row's
-- confirmation_count (a reliability signal, later useful for RAG ranking).
CREATE TABLE IF NOT EXISTS extracted_items (
    id                 BIGSERIAL PRIMARY KEY,
    session_id         BIGINT      NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    type               TEXT        NOT NULL,
    origin             TEXT        NOT NULL DEFAULT 'expert_claim',
    payload            JSONB       NOT NULL,
    quote              TEXT        NOT NULL,
    source_message_id  BIGINT      REFERENCES messages(id),
    embedding          vector(640),   -- harrier-oss-v1-270m hidden size
    duplicate_of       BIGINT      REFERENCES extracted_items(id),
    confirmation_count INT         NOT NULL DEFAULT 1,
    prompt_version     TEXT        NOT NULL DEFAULT 'v1',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_extracted_session ON extracted_items(session_id);
CREATE INDEX IF NOT EXISTS idx_extracted_type ON extracted_items(type);
CREATE INDEX IF NOT EXISTS idx_extracted_canonical
    ON extracted_items(duplicate_of);

-- Cross-session memory per topic. Regenerated from scratch each run from the
-- canonical extracted_items (duplicate_of IS NULL). The state_block gets a
-- trimmed excerpt (<=800 tokens); the full summary here may grow unbounded.
CREATE TABLE IF NOT EXISTS topic_summaries (
    topic          TEXT PRIMARY KEY,
    summary        TEXT        NOT NULL,
    prompt_version TEXT        NOT NULL DEFAULT 'v1',
    generated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Offline judge verdicts on agent questions. Statistics for prompt iteration,
-- not a production gate. expert_answer_message_id is the expert reply the
-- question was anchored to (the judge needs it to score anchored-vs-invented).
CREATE TABLE IF NOT EXISTS question_evals (
    id                       BIGSERIAL PRIMARY KEY,
    message_id               BIGINT      NOT NULL REFERENCES messages(id),
    expert_answer_message_id BIGINT      REFERENCES messages(id),
    verdict                  JSONB       NOT NULL,
    prompt_version           TEXT        NOT NULL DEFAULT 'v1',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
