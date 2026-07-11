-- Shared Telegram knowledge collection. CLI/legacy sessions remain private.
-- Run before claim backfill: moving already-projected claims requires an explicit
-- entity merge policy and must never happen silently.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM memory_claims c
        JOIN extracted_items e ON e.id=c.canonical_extracted_item_id
        JOIN sessions s ON s.id=e.session_id
        JOIN users u ON u.id=s.user_id
        WHERE u.telegram_user_id IS NOT NULL
    ) THEN
        RAISE EXCEPTION
          'public collection migration must run before Telegram claim backfill';
    END IF;
END $$;

INSERT INTO workspaces(owner_user_id,name,slug)
VALUES (NULL,'Public knowledge collection','public')
ON CONFLICT (slug) DO UPDATE SET name=EXCLUDED.name;

INSERT INTO workspace_members(workspace_id,user_id,role)
SELECT w.id,u.id,'member'
FROM workspaces w CROSS JOIN users u
WHERE w.slug='public' AND u.telegram_user_id IS NOT NULL
ON CONFLICT (workspace_id,user_id) DO UPDATE SET role=
    CASE WHEN workspace_members.role IN ('owner','admin')
         THEN workspace_members.role ELSE 'member' END;

INSERT INTO topics(workspace_id,name)
SELECT DISTINCT public.id,t.name
FROM sessions s
JOIN users u ON u.id=s.user_id AND u.telegram_user_id IS NOT NULL
JOIN topics t ON t.id=s.topic_id
CROSS JOIN LATERAL (SELECT id FROM workspaces WHERE slug='public') public
ON CONFLICT (workspace_id,name) DO NOTHING;

CREATE TEMP TABLE _public_topic_map ON COMMIT DROP AS
SELECT old.id AS old_topic_id,shared.id AS new_topic_id,shared.workspace_id
FROM topics old
JOIN sessions s ON s.topic_id=old.id
JOIN users u ON u.id=s.user_id AND u.telegram_user_id IS NOT NULL
JOIN workspaces public ON public.slug='public'
JOIN topics shared ON shared.workspace_id=public.id AND shared.name=old.name
GROUP BY old.id,shared.id,shared.workspace_id;

SET CONSTRAINTS ALL DEFERRED;

DELETE FROM topic_summaries ts
USING _public_topic_map map
WHERE ts.topic_id=map.old_topic_id;

UPDATE extracted_items e
SET workspace_id=map.workspace_id,topic_id=map.new_topic_id
FROM sessions s,_public_topic_map map
WHERE e.session_id=s.id AND s.topic_id=map.old_topic_id;

UPDATE postprocess_jobs j
SET workspace_id=map.workspace_id,topic_id=map.new_topic_id
FROM sessions s,_public_topic_map map
WHERE j.session_id=s.id AND s.topic_id=map.old_topic_id;

UPDATE sessions s
SET workspace_id=map.workspace_id,topic_id=map.new_topic_id
FROM _public_topic_map map
WHERE s.topic_id=map.old_topic_id
  AND EXISTS (
      SELECT 1 FROM users u
      WHERE u.id=s.user_id AND u.telegram_user_id IS NOT NULL
  );

SET CONSTRAINTS ALL IMMEDIATE;

-- Unused pre-claim graph is derived data. Drop rows rather than carry stale
-- tenant coordinates into shared memory.
DELETE FROM knowledge_edges;
DELETE FROM knowledge_nodes;

CREATE UNIQUE INDEX IF NOT EXISTS uq_sessions_workspace_id
    ON sessions(workspace_id,id);

ALTER TABLE sessions
    DROP CONSTRAINT IF EXISTS sessions_workspace_topic_fkey,
    ADD CONSTRAINT sessions_workspace_topic_fkey
      FOREIGN KEY (workspace_id,topic_id)
      REFERENCES topics(workspace_id,id),
    DROP CONSTRAINT IF EXISTS sessions_workspace_member_fkey,
    ADD CONSTRAINT sessions_workspace_member_fkey
      FOREIGN KEY (workspace_id,user_id)
      REFERENCES workspace_members(workspace_id,user_id);

ALTER TABLE extracted_items
    DROP CONSTRAINT IF EXISTS extracted_items_workspace_topic_fkey,
    ADD CONSTRAINT extracted_items_workspace_topic_fkey
      FOREIGN KEY (workspace_id,topic_id)
      REFERENCES topics(workspace_id,id);

ALTER TABLE postprocess_jobs
    DROP CONSTRAINT IF EXISTS postprocess_jobs_workspace_topic_fkey,
    ADD CONSTRAINT postprocess_jobs_workspace_topic_fkey
      FOREIGN KEY (workspace_id,topic_id)
      REFERENCES topics(workspace_id,id),
    DROP CONSTRAINT IF EXISTS postprocess_jobs_workspace_session_fkey,
    ADD CONSTRAINT postprocess_jobs_workspace_session_fkey
      FOREIGN KEY (workspace_id,session_id)
      REFERENCES sessions(workspace_id,id);
