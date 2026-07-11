-- RAPTOR-inspired aggregate memory. Claims/evidence remain source of truth.
CREATE TABLE IF NOT EXISTS memory_summary_nodes (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    level INT NOT NULL CHECK (level >= 0),
    cluster_key TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'current' CHECK (status IN ('current','stale')),
    prompt_version TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id,topic_id,level,cluster_key,prompt_version),
    UNIQUE (workspace_id,topic_id,id),
    FOREIGN KEY (workspace_id,topic_id) REFERENCES topics(workspace_id,id)
);
CREATE INDEX IF NOT EXISTS idx_memory_summary_current
    ON memory_summary_nodes(workspace_id,topic_id,level,id)
    WHERE status='current';

CREATE TABLE IF NOT EXISTS memory_summary_claims (
    workspace_id BIGINT NOT NULL,
    topic_id BIGINT NOT NULL,
    summary_id BIGINT NOT NULL,
    claim_id BIGINT NOT NULL,
    PRIMARY KEY (summary_id,claim_id),
    FOREIGN KEY (workspace_id,topic_id,summary_id)
      REFERENCES memory_summary_nodes(workspace_id,topic_id,id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id,topic_id,claim_id)
      REFERENCES memory_claims(workspace_id,topic_id,id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_summary_children (
    workspace_id BIGINT NOT NULL,
    topic_id BIGINT NOT NULL,
    parent_summary_id BIGINT NOT NULL,
    child_summary_id BIGINT NOT NULL,
    PRIMARY KEY (parent_summary_id,child_summary_id),
    CHECK (parent_summary_id <> child_summary_id),
    FOREIGN KEY (workspace_id,topic_id,parent_summary_id)
      REFERENCES memory_summary_nodes(workspace_id,topic_id,id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id,topic_id,child_summary_id)
      REFERENCES memory_summary_nodes(workspace_id,topic_id,id) ON DELETE CASCADE
);
