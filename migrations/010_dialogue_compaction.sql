-- Persistent, incremental dialogue compaction. Raw messages remain immutable;
-- this is only a regenerable prompt projection for long interviews.
CREATE TABLE IF NOT EXISTS session_context_compactions (
    session_id         BIGINT PRIMARY KEY
                       REFERENCES sessions(id) ON DELETE CASCADE,
    summary            TEXT NOT NULL,
    through_message_id BIGINT NOT NULL,
    prompt_version     TEXT NOT NULL,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT session_context_compaction_nonempty CHECK (btrim(summary) <> '')
);

CREATE INDEX IF NOT EXISTS idx_context_compaction_through
    ON session_context_compactions(session_id, through_message_id);

-- The cursor must point to a message from the same session. A trigger is used
-- instead of a plain FK because messages has no UNIQUE(session_id,id) yet.
CREATE OR REPLACE FUNCTION enforce_context_compaction_cursor()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM messages m
        WHERE m.id=NEW.through_message_id AND m.session_id=NEW.session_id
    ) THEN
        RAISE EXCEPTION 'context compaction cursor must belong to session';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_context_compaction_cursor
    ON session_context_compactions;
CREATE TRIGGER trg_context_compaction_cursor
BEFORE INSERT OR UPDATE OF session_id,through_message_id
ON session_context_compactions
FOR EACH ROW EXECUTE FUNCTION enforce_context_compaction_cursor();
