-- Migration 006 converted historical finished sessions to finalized before the
-- durable queue existed. Enqueue any such orphan exactly once.
INSERT INTO postprocess_jobs (
    workspace_id, session_id, topic_id, chat_id,
    extraction_version, prompt_version, model_version, idempotency_key
)
SELECT
    s.workspace_id, s.id, s.topic_id, NULL,
    'legacy-recovery', s.prompt_version, 'runtime-config',
    'postprocess:legacy-recovery:' || s.id
FROM sessions s
WHERE s.status='finalized'
  AND NOT EXISTS (
      SELECT 1 FROM postprocess_jobs j WHERE j.session_id=s.id
  )
ON CONFLICT (session_id) DO NOTHING;
