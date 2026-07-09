-- Stable Telegram identity for interview ownership.
--
-- expert_name is display metadata only: names are not unique and can change.
-- telegram_user_id is the durable key used by the bot to find the user's
-- active session. Legacy/script-created sessions may keep it NULL.

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS telegram_user_id BIGINT,
    ADD COLUMN IF NOT EXISTS telegram_username TEXT,
    ADD COLUMN IF NOT EXISTS telegram_full_name TEXT;

ALTER TABLE sessions
    DROP CONSTRAINT IF EXISTS sessions_telegram_user_id_positive,
    ADD CONSTRAINT sessions_telegram_user_id_positive
        CHECK (telegram_user_id IS NULL OR telegram_user_id > 0);

CREATE INDEX IF NOT EXISTS idx_sessions_telegram_user_id
    ON sessions(telegram_user_id)
    WHERE telegram_user_id IS NOT NULL;

-- One active interview per real Telegram user. This is a database invariant, so
-- two concurrent /start updates cannot create two active sessions for one user.
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_active_per_telegram_user
    ON sessions(telegram_user_id)
    WHERE telegram_user_id IS NOT NULL AND status = 'active';
