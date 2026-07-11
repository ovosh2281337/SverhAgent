-- Layered, tenant-safe long-term memory. PostgreSQL remains source of truth.

-- Historic pre-tenancy duplicate links could point across the new security
-- boundary. They are not meaningful after workspaces were split, so detach
-- them before installing the structural guardrail.
UPDATE extracted_items duplicate
SET duplicate_of = NULL
FROM extracted_items canonical
WHERE duplicate.duplicate_of = canonical.id
  AND (duplicate.workspace_id <> canonical.workspace_id
       OR duplicate.topic_id <> canonical.topic_id);

-- Flush deferred provenance/grounding checks raised by the repair before DDL
-- touches extracted_items in this same migration transaction.
SET CONSTRAINTS ALL IMMEDIATE;

CREATE UNIQUE INDEX IF NOT EXISTS uq_extracted_item_tenant_id
    ON extracted_items(workspace_id, topic_id, id);

ALTER TABLE extracted_items
    DROP CONSTRAINT IF EXISTS extracted_items_duplicate_of_fkey;
ALTER TABLE extracted_items
    DROP CONSTRAINT IF EXISTS extracted_items_tenant_duplicate_fkey;
ALTER TABLE extracted_items
    ADD CONSTRAINT extracted_items_tenant_duplicate_fkey
    FOREIGN KEY (workspace_id, topic_id, duplicate_of)
    REFERENCES extracted_items(workspace_id, topic_id, id)
    DEFERRABLE INITIALLY DEFERRED;

-- Confirmation means independent experts, not repeated sessions/overlap rows.
CREATE OR REPLACE FUNCTION recompute_item_confirmation_count(target_id BIGINT)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE extracted_items canonical
    SET confirmation_count = GREATEST(1, (
        SELECT count(DISTINCT evidence.user_id)
        FROM (
            SELECT s.user_id
            FROM extracted_items item
            JOIN sessions s ON s.id=item.session_id
            WHERE item.id=target_id
              AND item.grounding_status='verified'
            UNION ALL
            SELECT s.user_id
            FROM extracted_items item
            JOIN sessions s ON s.id=item.session_id
            WHERE item.duplicate_of=target_id
              AND item.grounding_status='verified'
              AND item.workspace_id=canonical.workspace_id
              AND item.topic_id=canonical.topic_id
        ) evidence
    ))
    WHERE canonical.id=target_id AND canonical.duplicate_of IS NULL;
END $$;

CREATE OR REPLACE FUNCTION refresh_item_confirmation_count()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    old_target BIGINT;
    new_target BIGINT;
BEGIN
    old_target := NULL;
    new_target := NULL;
    IF TG_OP IN ('DELETE', 'UPDATE') AND OLD.grounding_status='verified' THEN
        old_target := COALESCE(OLD.duplicate_of, OLD.id);
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') AND NEW.grounding_status='verified' THEN
        new_target := COALESCE(NEW.duplicate_of, NEW.id);
    END IF;
    IF old_target IS NOT NULL THEN
        PERFORM recompute_item_confirmation_count(old_target);
    END IF;
    IF new_target IS NOT NULL AND new_target IS DISTINCT FROM old_target THEN
        PERFORM recompute_item_confirmation_count(new_target);
    END IF;
    RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS trg_refresh_confirmation_count ON extracted_items;
CREATE TRIGGER trg_refresh_confirmation_count
AFTER INSERT OR DELETE OR UPDATE OF duplicate_of, grounding_status
ON extracted_items
FOR EACH ROW EXECUTE FUNCTION refresh_item_confirmation_count();

DO $$
DECLARE row RECORD;
BEGIN
    FOR row IN SELECT id FROM extracted_items WHERE duplicate_of IS NULL LOOP
        PERFORM recompute_item_confirmation_count(row.id);
    END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS memory_claims (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    canonical_extracted_item_id BIGINT NOT NULL,
    type TEXT NOT NULL,
    origin TEXT NOT NULL,
    normalized_statement TEXT NOT NULL,
    payload JSONB NOT NULL,
    embedding vector(640),
    embed_version TEXT NOT NULL,
    grounding_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'verified',
    version INT NOT NULL DEFAULT 1,
    observed_at TIMESTAMPTZ NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    superseded_by BIGINT,
    search_vector TSVECTOR GENERATED ALWAYS AS
      (to_tsvector('simple', normalized_statement)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_claim_status CHECK
      (status IN ('verified','needs_review','rejected','superseded')),
    CONSTRAINT memory_claim_validity CHECK
      (valid_to IS NULL OR valid_to >= valid_from),
    UNIQUE (workspace_id, topic_id, canonical_extracted_item_id),
    UNIQUE (workspace_id, id),
    UNIQUE (workspace_id, topic_id, id),
    FOREIGN KEY (workspace_id, topic_id)
      REFERENCES topics(workspace_id, id),
    FOREIGN KEY (workspace_id, topic_id, canonical_extracted_item_id)
      REFERENCES extracted_items(workspace_id, topic_id, id),
    FOREIGN KEY (workspace_id, topic_id, superseded_by)
      REFERENCES memory_claims(workspace_id, topic_id, id)
      DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX IF NOT EXISTS idx_memory_claims_vector
    ON memory_claims USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL AND status='verified';
CREATE INDEX IF NOT EXISTS idx_memory_claims_fts
    ON memory_claims USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_memory_claims_current
    ON memory_claims(workspace_id, topic_id, status, id)
    WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS memory_claim_evidence (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL,
    topic_id BIGINT NOT NULL,
    claim_id BIGINT NOT NULL,
    extracted_item_id BIGINT NOT NULL,
    session_id BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (claim_id, extracted_item_id),
    FOREIGN KEY (workspace_id, topic_id, claim_id)
      REFERENCES memory_claims(workspace_id, topic_id, id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id, topic_id, extracted_item_id)
      REFERENCES extracted_items(workspace_id, topic_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_distinct_users
    ON memory_claim_evidence(claim_id, user_id);

CREATE TABLE IF NOT EXISTS memory_claim_relations (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL,
    source_claim_id BIGINT NOT NULL,
    target_claim_id BIGINT NOT NULL,
    relation_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'needs_review',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    classifier_version TEXT NOT NULL,
    verifier_version TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_relation_type CHECK
      (relation_type IN ('duplicate_of','supports','contradicts','refines','depends_on')),
    CONSTRAINT memory_relation_status CHECK
      (status IN ('verified','needs_review','rejected','superseded')),
    CONSTRAINT memory_relation_not_self CHECK (source_claim_id <> target_claim_id),
    UNIQUE (workspace_id, source_claim_id, target_claim_id, relation_type,
            classifier_version, verifier_version),
    FOREIGN KEY (workspace_id, source_claim_id)
      REFERENCES memory_claims(workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id, target_claim_id)
      REFERENCES memory_claims(workspace_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memory_relations_source
    ON memory_claim_relations(workspace_id, source_claim_id, status, relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_relations_target
    ON memory_claim_relations(workspace_id, target_claim_id, status, relation_type);

CREATE TABLE IF NOT EXISTS memory_entities (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    normalized_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    embedding vector(640),
    embed_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'verified',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_entity_status CHECK
      (status IN ('verified','needs_review','rejected','superseded')),
    UNIQUE (workspace_id, normalized_name),
    UNIQUE (workspace_id, id)
);
CREATE TABLE IF NOT EXISTS memory_entity_aliases (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL,
    entity_id BIGINT NOT NULL,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, entity_id, normalized_alias),
    FOREIGN KEY (workspace_id, entity_id)
      REFERENCES memory_entities(workspace_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memory_entity_alias_lookup
    ON memory_entity_aliases(workspace_id, normalized_alias);

CREATE TABLE IF NOT EXISTS memory_claim_entities (
    workspace_id BIGINT NOT NULL,
    claim_id BIGINT NOT NULL,
    entity_id BIGINT NOT NULL,
    mention TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (claim_id, entity_id, mention),
    FOREIGN KEY (workspace_id, claim_id)
      REFERENCES memory_claims(workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id, entity_id)
      REFERENCES memory_entities(workspace_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_shadow_logs (
    id BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    session_id BIGINT REFERENCES sessions(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    basic_item_ids BIGINT[] NOT NULL DEFAULT '{}',
    hybrid_claim_ids BIGINT[] NOT NULL DEFAULT '{}',
    basic_latency_ms DOUBLE PRECISION NOT NULL,
    hybrid_latency_ms DOUBLE PRECISION NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (workspace_id, topic_id) REFERENCES topics(workspace_id, id)
);
