from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src import bot, config, db


class _FakePool:
    def __init__(self) -> None:
        self.fetchrow = AsyncMock(return_value=None)


class SessionIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_session_uses_telegram_user_id_not_display_name(self) -> None:
        fake = _FakePool()
        with patch.object(db, "pool", AsyncMock(return_value=fake)):
            await db.active_session_for_user(123456)

        sql = fake.fetchrow.await_args.args[0]
        self.assertIn("telegram_user_id=$1", sql)
        self.assertNotIn("expert_name=$1", sql)
        self.assertEqual(fake.fetchrow.await_args.args[1], 123456)

    async def test_create_session_persists_telegram_identity_metadata(self) -> None:
        fake = _FakePool()
        with patch.object(db, "pool", AsyncMock(return_value=fake)):
            await db.create_session(
                "Same Name",
                "topic",
                telegram_user_id=42,
                telegram_username="same_name",
                telegram_full_name="Same Name",
                user_id=7,
                workspace_id=8,
                topic_id=9,
            )

        sql = fake.fetchrow.await_args.args[0]
        args = fake.fetchrow.await_args.args
        self.assertIn("telegram_user_id", sql)
        self.assertIn("telegram_username", sql)
        self.assertIn("telegram_full_name", sql)
        self.assertEqual(args[1], "Same Name")
        self.assertEqual(args[2], "topic")
        self.assertEqual(args[3], config.PROMPT_VERSION)
        self.assertEqual(args[4], 42)
        self.assertEqual(args[5], "same_name")
        self.assertEqual(args[6], "Same Name")
        self.assertEqual(args[7:10], (7, 8, 9))

    async def test_invalid_telegram_user_id_is_rejected_before_db(self) -> None:
        fake = _FakePool()
        with patch.object(db, "pool", AsyncMock(return_value=fake)):
            with self.assertRaises(ValueError):
                await db.active_session_for_user(0)
            with self.assertRaises(ValueError):
                await db.create_session("x", "topic", telegram_user_id=-1)
        fake.fetchrow.assert_not_called()

    def test_bot_identity_helpers_use_user_id_and_keep_name_as_label(self) -> None:
        msg = SimpleNamespace(
            from_user=SimpleNamespace(
                id=9001,
                username="expert_login",
                full_name="Same Display Name",
            )
        )

        self.assertEqual(bot._telegram_user_id(msg), 9001)
        self.assertEqual(bot._telegram_username(msg), "expert_login")
        self.assertEqual(bot._telegram_full_name(msg), "Same Display Name")
        self.assertEqual(bot._expert_name(msg), "Same Display Name")

    def test_migration_enforces_one_active_session_per_telegram_user(self) -> None:
        sql = Path("migrations/005_telegram_identity.sql").read_text(encoding="utf-8")
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS", sql)
        self.assertIn("ON sessions(telegram_user_id)", sql)
        self.assertIn("WHERE telegram_user_id IS NOT NULL AND status = 'active'", sql)


if __name__ == "__main__":
    unittest.main()
