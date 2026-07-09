-- Memory is publish-on-complete: neither RAG nor topic summaries may observe a
-- session while extraction is still writing its derived rows. Application
-- queries also require config.EMBED_TEXT_VERSION; these indexes support the
-- stable status/version filters without indexing unfinished/duplicate rows.
CREATE INDEX IF NOT EXISTS idx_sessions_extracted_topic
    ON sessions(topic, id) WHERE status = 'extracted';

CREATE INDEX IF NOT EXISTS idx_extracted_published_version
    ON extracted_items(session_id, embed_version, id)
    WHERE duplicate_of IS NULL;

-- Quarantine is deletion here because extracted_items is a regenerable derived
-- layer and messages remains the immutable source of truth. Remove legacy rows
-- whose citation cannot be proven, plus rows depending on an invalid canonical
-- item. Valid sessions can be explicitly re-extracted from their transcript.
WITH RECURSIVE invalid(id) AS (
    SELECT e.id
    FROM extracted_items e
    LEFT JOIN messages m ON m.id = e.source_message_id
    WHERE e.type NOT IN ('fact', 'qa_pair', 'term')
       OR e.origin NOT IN ('expert_claim', 'confirmed_hypothesis')
       OR jsonb_typeof(e.payload) IS DISTINCT FROM 'object'
       OR e.source_message_id IS NULL
       OR m.id IS NULL
       OR m.session_id <> e.session_id
       OR m.role <> 'user'
       OR btrim(e.quote) = ''
       OR strpos(m.content, e.quote) = 0
    UNION
    SELECT child.id
    FROM extracted_items child
    JOIN invalid parent ON child.duplicate_of = parent.id
)
DELETE FROM extracted_items e
USING invalid
WHERE e.id = invalid.id;

-- Legacy invalid duplicates may have inflated this denormalized counter before
-- being removed. Rebuild it from the remaining provenance rows.
UPDATE extracted_items canonical
SET confirmation_count = 1 + (
    SELECT count(*) FROM extracted_items duplicate
    WHERE duplicate.duplicate_of = canonical.id
)
WHERE canonical.duplicate_of IS NULL;

-- Structural guardrails for all future writes. Cross-table grounding (same
-- session, expert role, exact substring) is additionally enforced by the
-- INSERT ... SELECT boundary in db.add_extracted_item.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_type_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_type_check
            CHECK (type IN ('fact', 'qa_pair', 'term'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_origin_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_origin_check
            CHECK (origin IN ('expert_claim', 'confirmed_hypothesis'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_grounding_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_grounding_check
            CHECK (
                source_message_id IS NOT NULL
                AND btrim(quote) <> ''
                AND jsonb_typeof(payload) = 'object'
            );
    END IF;
END $$;

-- Existing summaries may already contain rows that are now unpublished or use
-- an incompatible embedding text version. They are derived and will be rebuilt
-- after the next successful extraction for each topic.
DELETE FROM topic_summaries;
