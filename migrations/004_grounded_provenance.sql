-- Provenance v1: separate "where text came from" from "whether it entails the
-- normalized claim". Existing rows remain auditable but are deliberately
-- marked legacy; only a new grounding pass may publish them again.

ALTER TABLE extracted_items
    ADD COLUMN IF NOT EXISTS support_mode TEXT,
    ADD COLUMN IF NOT EXISTS grounding_status TEXT NOT NULL DEFAULT 'legacy',
    ADD COLUMN IF NOT EXISTS grounding_version TEXT NOT NULL DEFAULT 'legacy',
    ADD COLUMN IF NOT EXISTS grounding_details JSONB NOT NULL DEFAULT '{}'::jsonb;

-- A span is zero-based and end-exclusive in Unicode characters, matching
-- Python string slicing. The quoted text is reconstructed from messages, which
-- is the immutable source of truth, rather than copied into this table.
CREATE TABLE IF NOT EXISTS extracted_item_provenance (
    id          BIGSERIAL PRIMARY KEY,
    item_id     BIGINT NOT NULL REFERENCES extracted_items(id) ON DELETE CASCADE,
    message_id  BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    kind        TEXT   NOT NULL,
    start_char  INT    NOT NULL,
    end_char    INT    NOT NULL,
    ord         INT    NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT extracted_item_provenance_kind_check CHECK (
        kind IN ('user_support', 'question_context', 'hypothesis_target',
                 'confirmation', 'correction_target')
    ),
    CONSTRAINT extracted_item_provenance_offsets_check CHECK (
        start_char >= 0 AND end_char > start_char
    ),
    CONSTRAINT extracted_item_provenance_ord_check CHECK (ord >= 0),
    CONSTRAINT extracted_item_provenance_item_ord_key UNIQUE (item_id, ord),
    CONSTRAINT extracted_item_provenance_span_key UNIQUE (
        item_id, message_id, kind, start_char, end_char
    )
);

CREATE INDEX IF NOT EXISTS idx_provenance_message
    ON extracted_item_provenance(message_id, item_id);

-- Invalid model output that cannot be represented as an item/provenance graph
-- must remain inspectable. It is not eligible for RAG or summaries.
CREATE TABLE IF NOT EXISTS extraction_rejections (
    id                 BIGSERIAL PRIMARY KEY,
    session_id         BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    stage              TEXT   NOT NULL,
    reason             TEXT   NOT NULL,
    raw_item           JSONB  NOT NULL,
    attempted_repair   BOOLEAN NOT NULL DEFAULT false,
    grounding_version  TEXT   NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT extraction_rejections_stage_check CHECK (
        stage IN ('validation', 'grounding', 'repair')
    ),
    CONSTRAINT extraction_rejections_raw_check CHECK (
        jsonb_typeof(raw_item) IN ('object', 'array')
    )
);

CREATE INDEX IF NOT EXISTS idx_extraction_rejections_session
    ON extraction_rejections(session_id, grounding_version, id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_support_mode_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_support_mode_check
            CHECK (support_mode IS NULL OR support_mode IN (
                'direct_assertion', 'contextual_answer',
                'explicit_confirmation', 'correction', 'multi_turn_synthesis'
            ));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_grounding_status_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_grounding_status_check
            CHECK (grounding_status IN (
                'verified', 'partial', 'needs_review', 'legacy'
            ));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_grounding_details_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_grounding_details_check
            CHECK (jsonb_typeof(grounding_details) = 'object');
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_grounded_shape_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_grounded_shape_check
            CHECK (
                grounding_status = 'legacy'
                OR (support_mode IS NOT NULL AND grounding_version <> 'legacy')
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'extracted_items_not_self_duplicate_check'
          AND conrelid = 'extracted_items'::regclass
    ) THEN
        ALTER TABLE extracted_items ADD CONSTRAINT extracted_items_not_self_duplicate_check
            CHECK (duplicate_of IS NULL OR duplicate_of <> id);
    END IF;
END $$;

-- Preserve the old single quote as an auditable span, but do not claim that it
-- semantically supports the payload. The legacy status keeps it unpublished.
INSERT INTO extracted_item_provenance
    (item_id, message_id, kind, start_char, end_char, ord)
SELECT e.id, e.source_message_id, 'user_support',
       strpos(m.content, e.quote) - 1,
       strpos(m.content, e.quote) - 1 + char_length(e.quote),
       0
FROM extracted_items e
JOIN messages m ON m.id = e.source_message_id
WHERE e.grounding_status = 'legacy'
  AND strpos(m.content, e.quote) > 0
ON CONFLICT DO NOTHING;

-- Validate every span at the database boundary. Role-bearing kinds cannot be
-- swapped, and an offset cannot point outside the immutable source message.
CREATE OR REPLACE FUNCTION validate_provenance_span()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    item_session BIGINT;
    message_session BIGINT;
    message_role TEXT;
    message_length INT;
BEGIN
    SELECT session_id INTO item_session
    FROM extracted_items WHERE id = NEW.item_id;

    SELECT session_id, role, char_length(content)
    INTO message_session, message_role, message_length
    FROM messages WHERE id = NEW.message_id;

    IF item_session IS NULL OR message_session IS NULL OR item_session <> message_session THEN
        RAISE EXCEPTION 'provenance item/message must belong to the same session';
    END IF;
    IF NEW.end_char > message_length THEN
        RAISE EXCEPTION 'provenance offset % exceeds message length %',
            NEW.end_char, message_length;
    END IF;
    IF NEW.kind IN ('user_support', 'confirmation') AND message_role <> 'user' THEN
        RAISE EXCEPTION 'provenance kind % requires a user message', NEW.kind;
    END IF;
    IF NEW.kind IN ('question_context', 'hypothesis_target')
       AND message_role <> 'assistant' THEN
        RAISE EXCEPTION 'provenance kind % requires an assistant message', NEW.kind;
    END IF;
    IF NEW.kind = 'correction_target' AND message_role NOT IN ('user', 'assistant') THEN
        RAISE EXCEPTION 'correction_target requires a user or assistant message';
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS trg_validate_provenance_span
    ON extracted_item_provenance;
CREATE TRIGGER trg_validate_provenance_span
BEFORE INSERT OR UPDATE ON extracted_item_provenance
FOR EACH ROW EXECUTE FUNCTION validate_provenance_span();

-- Deferred validation lets the application insert the item first and all of
-- its spans second inside one short transaction. COMMIT succeeds only when the
-- complete graph satisfies the support-mode contract and temporal ordering.
CREATE OR REPLACE FUNCTION validate_extracted_item_grounding()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    checked_item BIGINT;
    item_row extracted_items%ROWTYPE;
    total_count INT;
    user_count INT;
    user_messages INT;
    context_count INT;
    hypothesis_count INT;
    confirmation_count_local INT;
    correction_count INT;
    bad_order_count INT;
    duplicate_status TEXT;
    duplicate_parent BIGINT;
BEGIN
    IF TG_TABLE_NAME = 'extracted_items' THEN
        IF TG_OP = 'DELETE' THEN
            checked_item := OLD.id;
        ELSE
            checked_item := NEW.id;
        END IF;
    ELSE
        IF TG_OP = 'DELETE' THEN
            checked_item := OLD.item_id;
        ELSE
            checked_item := NEW.item_id;
        END IF;
    END IF;
    SELECT * INTO item_row FROM extracted_items WHERE id = checked_item;
    IF NOT FOUND OR item_row.grounding_status = 'legacy' THEN
        RETURN NULL;
    END IF;

    SELECT count(*),
           count(*) FILTER (WHERE kind = 'user_support'),
           count(DISTINCT message_id) FILTER (WHERE kind = 'user_support'),
           count(*) FILTER (WHERE kind = 'question_context'),
           count(*) FILTER (WHERE kind = 'hypothesis_target'),
           count(*) FILTER (WHERE kind = 'confirmation'),
           count(*) FILTER (WHERE kind = 'correction_target')
    INTO total_count, user_count, user_messages, context_count,
         hypothesis_count, confirmation_count_local, correction_count
    FROM extracted_item_provenance WHERE item_id = checked_item;

    IF item_row.support_mode = 'direct_assertion' THEN
        IF user_count < 1 OR total_count <> user_count
           OR item_row.origin <> 'expert_claim' THEN
            RAISE EXCEPTION 'invalid direct_assertion provenance for item %', checked_item;
        END IF;
    ELSIF item_row.support_mode = 'contextual_answer' THEN
        IF user_count < 1 OR context_count < 1
           OR total_count <> user_count + context_count
           OR item_row.origin <> 'expert_claim' THEN
            RAISE EXCEPTION 'invalid contextual_answer provenance for item %', checked_item;
        END IF;
        SELECT count(*) INTO bad_order_count
        FROM extracted_item_provenance context_span
        WHERE context_span.item_id = checked_item
          AND context_span.kind = 'question_context'
          AND NOT EXISTS (
              SELECT 1 FROM extracted_item_provenance support_span
              WHERE support_span.item_id = checked_item
                AND support_span.kind = 'user_support'
                AND support_span.message_id > context_span.message_id
          );
        IF bad_order_count > 0 THEN
            RAISE EXCEPTION 'question_context must precede user_support for item %', checked_item;
        END IF;
    ELSIF item_row.support_mode = 'explicit_confirmation' THEN
        IF hypothesis_count < 1 OR confirmation_count_local < 1
           OR total_count <> hypothesis_count + confirmation_count_local
           OR item_row.origin <> 'confirmed_hypothesis' THEN
            RAISE EXCEPTION 'invalid explicit_confirmation provenance for item %', checked_item;
        END IF;
        SELECT count(*) INTO bad_order_count
        FROM extracted_item_provenance target_span
        WHERE target_span.item_id = checked_item
          AND target_span.kind = 'hypothesis_target'
          AND NOT EXISTS (
              SELECT 1 FROM extracted_item_provenance confirmation_span
              WHERE confirmation_span.item_id = checked_item
                AND confirmation_span.kind = 'confirmation'
                AND confirmation_span.message_id > target_span.message_id
          );
        IF bad_order_count > 0 THEN
            RAISE EXCEPTION 'hypothesis_target must precede confirmation for item %', checked_item;
        END IF;
    ELSIF item_row.support_mode = 'correction' THEN
        IF user_count < 1 OR correction_count < 1
           OR total_count <> user_count + correction_count
           OR item_row.origin <> 'expert_claim' THEN
            RAISE EXCEPTION 'invalid correction provenance for item %', checked_item;
        END IF;
        SELECT count(*) INTO bad_order_count
        FROM extracted_item_provenance target_span
        WHERE target_span.item_id = checked_item
          AND target_span.kind = 'correction_target'
          AND NOT EXISTS (
              SELECT 1 FROM extracted_item_provenance support_span
              WHERE support_span.item_id = checked_item
                AND support_span.kind = 'user_support'
                AND support_span.message_id > target_span.message_id
          );
        IF bad_order_count > 0 THEN
            RAISE EXCEPTION 'correction_target must precede user_support for item %', checked_item;
        END IF;
    ELSIF item_row.support_mode = 'multi_turn_synthesis' THEN
        IF user_messages < 2 OR total_count <> user_count + context_count
           OR item_row.origin <> 'expert_claim' THEN
            RAISE EXCEPTION 'invalid multi_turn_synthesis provenance for item %', checked_item;
        END IF;
    ELSE
        RAISE EXCEPTION 'grounded item % has unknown support_mode %',
            checked_item, item_row.support_mode;
    END IF;

    IF item_row.duplicate_of IS NOT NULL THEN
        SELECT grounding_status, duplicate_of
        INTO duplicate_status, duplicate_parent
        FROM extracted_items WHERE id = item_row.duplicate_of;
        IF duplicate_status IS DISTINCT FROM 'verified' OR duplicate_parent IS NOT NULL THEN
            RAISE EXCEPTION 'duplicate item % must target a verified canonical item', checked_item;
        END IF;
    END IF;
    RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS trg_validate_grounded_item ON extracted_items;
CREATE CONSTRAINT TRIGGER trg_validate_grounded_item
AFTER INSERT OR UPDATE OF support_mode, grounding_status, grounding_version,
    origin, duplicate_of ON extracted_items
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION validate_extracted_item_grounding();

DROP TRIGGER IF EXISTS trg_validate_grounded_provenance
    ON extracted_item_provenance;
CREATE CONSTRAINT TRIGGER trg_validate_grounded_provenance
AFTER INSERT OR UPDATE OR DELETE ON extracted_item_provenance
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION validate_extracted_item_grounding();

-- confirmation_count is derived from committed verified duplicate rows. The
-- trigger runs in the same transaction as item + provenance; a deferred
-- grounding failure rolls both the duplicate and this update back.
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

    IF TG_OP IN ('DELETE', 'UPDATE') AND OLD.grounding_status = 'verified' THEN
        old_target := OLD.duplicate_of;
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') AND NEW.grounding_status = 'verified' THEN
        new_target := NEW.duplicate_of;
    END IF;

    IF old_target IS NOT NULL THEN
        UPDATE extracted_items canonical
        SET confirmation_count = 1 + (
            SELECT count(*) FROM extracted_items duplicate
            WHERE duplicate.duplicate_of = old_target
              AND duplicate.grounding_status = 'verified'
        )
        WHERE canonical.id = old_target AND canonical.duplicate_of IS NULL;
    END IF;
    IF new_target IS NOT NULL AND new_target IS DISTINCT FROM old_target THEN
        UPDATE extracted_items canonical
        SET confirmation_count = 1 + (
            SELECT count(*) FROM extracted_items duplicate
            WHERE duplicate.duplicate_of = new_target
              AND duplicate.grounding_status = 'verified'
        )
        WHERE canonical.id = new_target AND canonical.duplicate_of IS NULL;
    END IF;
    RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS trg_refresh_confirmation_count ON extracted_items;
CREATE TRIGGER trg_refresh_confirmation_count
AFTER INSERT OR DELETE OR UPDATE OF duplicate_of, grounding_status
ON extracted_items
FOR EACH ROW EXECUTE FUNCTION refresh_item_confirmation_count();

CREATE INDEX IF NOT EXISTS idx_extracted_verified_memory
    ON extracted_items(session_id, grounding_version, embed_version, id)
    WHERE grounding_status = 'verified' AND duplicate_of IS NULL;

CREATE INDEX IF NOT EXISTS idx_extracted_verified_duplicates
    ON extracted_items(duplicate_of, id)
    WHERE grounding_status = 'verified' AND duplicate_of IS NOT NULL;

-- All rows that predate this migration are provenance-auditable but not
-- semantically verified. Re-grounding creates new g1 rows without deleting
-- these records, so a failed reprocess never destroys the old derived layer.
-- The `WHERE grounding_version = 'legacy'` guard makes this idempotent: a manual
-- re-run must NOT drag already re-grounded (g1) verified rows back to legacy.
UPDATE extracted_items
SET support_mode = NULL,
    grounding_status = 'legacy',
    grounding_version = 'legacy',
    grounding_details = jsonb_build_object(
        'reason', 'predates semantic grounding; exact quote proves location only'
    ),
    confirmation_count = CASE WHEN duplicate_of IS NULL THEN 1 ELSE confirmation_count END
WHERE grounding_version = 'legacy';

-- Summaries may contain now-legacy claims; drop them so they are rebuilt only
-- from the new verified layer. Guard makes this idempotent: once ANY row has been
-- re-grounded, a re-run of this migration must not wipe rebuilt summaries.
DELETE FROM topic_summaries
WHERE NOT EXISTS (
    SELECT 1 FROM extracted_items WHERE grounding_version <> 'legacy'
);
