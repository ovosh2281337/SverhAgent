-- Track which _embed_text() format produced each row's vector, so a format
-- change (see config.EMBED_TEXT_VERSION) can re-embed only outdated rows instead
-- of the whole base. Existing rows were re-embedded in the v2 plain-prose format
-- on 2026-07-06, so default them to 'v2'.
ALTER TABLE extracted_items
    ADD COLUMN IF NOT EXISTS embed_version TEXT NOT NULL DEFAULT 'v2';
