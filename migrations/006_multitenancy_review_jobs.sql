-- Security boundary, review workflow, and durable post-processing.

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    telegram_user_id BIGINT UNIQUE,
    telegram_username TEXT,
    telegram_full_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT users_telegram_id_positive
        CHECK (telegram_user_id IS NULL OR telegram_user_id > 0)
);

CREATE TABLE IF NOT EXISTS workspaces (
    id BIGSERIAL PRIMARY KEY,
    owner_user_id BIGINT UNIQUE REFERENCES users(id),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workspace_members (
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, user_id),
    CONSTRAINT workspace_member_role
        CHECK (role IN ('owner', 'admin', 'member'))
);

CREATE TABLE IF NOT EXISTS topics (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, name),
    UNIQUE (workspace_id, id)
);

-- Backfill every historic Telegram identity. Each NULL identity gets an
-- isolated legacy workspace; CLI-created sessions never become mutually visible
-- and are not exposed to any Telegram user.
INSERT INTO users (telegram_user_id, telegram_username, telegram_full_name)
SELECT DISTINCT telegram_user_id, telegram_username, telegram_full_name
FROM sessions
WHERE telegram_user_id IS NOT NULL
ON CONFLICT (telegram_user_id) DO UPDATE SET
    telegram_username = COALESCE(EXCLUDED.telegram_username, users.telegram_username),
    telegram_full_name = COALESCE(EXCLUDED.telegram_full_name, users.telegram_full_name),
    updated_at = now();

INSERT INTO users (telegram_user_id, telegram_username, telegram_full_name)
SELECT NULL, 'legacy-session-' || s.id::text, 'Legacy session ' || s.id::text
FROM sessions s
WHERE s.telegram_user_id IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM users u WHERE u.telegram_username='legacy-session-' || s.id::text
  );

INSERT INTO workspaces (owner_user_id, name, slug)
SELECT id, COALESCE(telegram_full_name, telegram_username, 'Personal workspace'),
       'personal-' || id::text
FROM users
ON CONFLICT (owner_user_id) DO NOTHING;

INSERT INTO workspace_members (workspace_id, user_id, role)
SELECT w.id, w.owner_user_id, 'owner'
FROM workspaces w
WHERE w.owner_user_id IS NOT NULL
ON CONFLICT (workspace_id, user_id) DO UPDATE SET role='owner';

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id),
    ADD COLUMN IF NOT EXISTS workspace_id BIGINT REFERENCES workspaces(id),
    ADD COLUMN IF NOT EXISTS topic_id BIGINT REFERENCES topics(id);

UPDATE sessions s
SET user_id = u.id
FROM users u
WHERE s.user_id IS NULL
  AND ((s.telegram_user_id IS NOT NULL AND u.telegram_user_id=s.telegram_user_id)
       OR (s.telegram_user_id IS NULL
           AND u.telegram_username='legacy-session-' || s.id::text));

UPDATE sessions s
SET workspace_id = w.id
FROM workspaces w
WHERE s.workspace_id IS NULL AND w.owner_user_id=s.user_id;

INSERT INTO topics (workspace_id, name)
SELECT DISTINCT workspace_id, topic FROM sessions
WHERE workspace_id IS NOT NULL
ON CONFLICT (workspace_id, name) DO NOTHING;

UPDATE sessions s
SET topic_id=t.id
FROM topics t
WHERE s.topic_id IS NULL
  AND t.workspace_id=s.workspace_id AND t.name=s.topic;

ALTER TABLE sessions
    ALTER COLUMN user_id SET NOT NULL,
    ALTER COLUMN workspace_id SET NOT NULL,
    ALTER COLUMN topic_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_workspace_topic
    ON sessions(workspace_id, topic_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_status
    ON sessions(user_id, status, id DESC);

UPDATE sessions SET status='finalized' WHERE status='finished';

DROP INDEX IF EXISTS idx_sessions_one_active_per_telegram_user;
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_open_per_user
    ON sessions(user_id)
    WHERE status IN ('active', 'draft_review');

-- Review never mutates raw message content. Approved snapshots are appended as
-- new immutable messages and become extraction input.
ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS included_in_extraction BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS review_items (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_message_id BIGINT REFERENCES messages(id),
    ord INT NOT NULL,
    text TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_review_items_session
    ON review_items(session_id, ord) WHERE NOT deleted;

-- Old topic summaries were keyed only by user-controlled text and may combine
-- tenants. They are derived data: discard and rebuild inside workspace/topic.
DROP TABLE IF EXISTS topic_summaries;
CREATE TABLE topic_summaries (
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    prompt_version TEXT NOT NULL DEFAULT 'v1',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, topic_id),
    FOREIGN KEY (workspace_id, topic_id) REFERENCES topics(workspace_id, id)
);

ALTER TABLE extracted_items
    ADD COLUMN IF NOT EXISTS workspace_id BIGINT REFERENCES workspaces(id),
    ADD COLUMN IF NOT EXISTS topic_id BIGINT REFERENCES topics(id);
UPDATE extracted_items e
SET workspace_id=s.workspace_id, topic_id=s.topic_id
FROM sessions s
WHERE e.session_id=s.id AND (e.workspace_id IS NULL OR e.topic_id IS NULL);
ALTER TABLE extracted_items
    ALTER COLUMN workspace_id SET NOT NULL,
    ALTER COLUMN topic_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS idx_extracted_workspace_topic
    ON extracted_items(workspace_id, topic_id, id);

CREATE TABLE IF NOT EXISTS postprocess_jobs (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    session_id BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    chat_id BIGINT,
    status TEXT NOT NULL DEFAULT 'queued',
    extraction_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 5,
    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_expires_at TIMESTAMPTZ,
    worker_id TEXT,
    last_error TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT postprocess_job_status CHECK
        (status IN ('queued','running','retry_wait','succeeded','dead')),
    UNIQUE (session_id)
);
CREATE INDEX IF NOT EXISTS idx_postprocess_jobs_ready
    ON postprocess_jobs(status, available_at, id)
    WHERE status IN ('queued', 'retry_wait');

-- Future graph tables already carry tenant boundary. LLM output enters only as
-- candidate; publication requires a later verifier.
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    extracted_item_id BIGINT REFERENCES extracted_items(id),
    statement TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_node_status CHECK
      (status IN ('candidate','verified','needs_review','rejected','superseded')),
    UNIQUE (workspace_id, id),
    FOREIGN KEY (workspace_id, topic_id) REFERENCES topics(workspace_id, id)
);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_workspace_topic
    ON knowledge_nodes(workspace_id, topic_id, id);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    source_node_id BIGINT NOT NULL,
    target_node_id BIGINT NOT NULL,
    relation_type TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'candidate',
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_edge_type CHECK (relation_type IN
      ('duplicate_of','supports','contradicts','related_to','refines',
       'depends_on','stated_by','derived_from')),
    CONSTRAINT knowledge_edge_status CHECK
      (status IN ('candidate','verified','needs_review','rejected','superseded')),
    CONSTRAINT knowledge_edge_not_self CHECK (source_node_id <> target_node_id),
    FOREIGN KEY (workspace_id, topic_id) REFERENCES topics(workspace_id, id),
    FOREIGN KEY (workspace_id, source_node_id)
      REFERENCES knowledge_nodes(workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id, target_node_id)
      REFERENCES knowledge_nodes(workspace_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_workspace_topic
    ON knowledge_edges(workspace_id, topic_id, source_node_id, target_node_id);
