from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src import bot, db
from src.jobs import worker


class _Msg:
    def __init__(self, *, user_id: int = 101, chat_type: str = "private") -> None:
        self.from_user = SimpleNamespace(
            id=user_id, username=f"u{user_id}", full_name=f"User {user_id}"
        )
        self.chat = SimpleNamespace(id=user_id + 1000, type=chat_type)
        self.answer = AsyncMock()


class _Pool:
    def __init__(self) -> None:
        self.fetchrow = AsyncMock(return_value=None)


class MultitenancyTests(unittest.IsolatedAsyncioTestCase):
    async def test_membership_lookup_cannot_cross_workspace(self) -> None:
        fake = _Pool()
        with patch.object(db, "pool", AsyncMock(return_value=fake)):
            await db.workspace_access_for_user(101, 9002)
        sql = fake.fetchrow.await_args.args[0]
        self.assertIn("workspace_members", sql)
        self.assertIn("u.telegram_user_id=$1", sql)
        self.assertIn("wm.workspace_id=$2", sql)
        self.assertEqual(fake.fetchrow.await_args.args[1:], (101, 9002))

    async def test_start_resolves_same_topic_only_inside_callers_workspace(self) -> None:
        msg = _Msg(user_id=101)
        command = SimpleNamespace(args="same-name")
        with (
            patch.object(db, "open_session_for_user", AsyncMock(return_value=None)),
            patch.object(
                db, "ensure_user_workspace",
                AsyncMock(return_value={"user_id": 7, "workspace_id": 70, "role": "owner"}),
            ),
            patch.object(
                db, "resolve_topic",
                AsyncMock(return_value={"id": 700, "name": "same-name"}),
            ) as resolve,
            patch.object(db, "create_session", AsyncMock()) as create,
        ):
            await bot.on_start(msg, command)
        resolve.assert_awaited_once_with(70, "same-name")
        self.assertEqual(create.await_args.kwargs["workspace_id"], 70)
        self.assertEqual(create.await_args.kwargs["topic_id"], 700)

    async def test_verbose_is_denied_to_non_admin(self) -> None:
        msg = _Msg()
        bot._verbose.clear()
        with patch.object(
            db, "ensure_user_workspace",
            AsyncMock(return_value={"user_id": 7, "workspace_id": 70, "role": "member"}),
        ):
            await bot.on_verbose(msg)
        self.assertNotIn(msg.chat.id, bot._verbose)
        self.assertIn("только администратору", msg.answer.await_args.args[0])

    async def test_group_chat_is_explicitly_rejected(self) -> None:
        msg = _Msg(chat_type="group")
        await bot.reject_group_chat(msg)
        self.assertIn("Групповые чаты отключены", msg.answer.await_args.args[0])


class ReviewAndJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_approve_enqueues_but_does_not_run_inline(self) -> None:
        msg = _Msg()
        session = {
            "id": 9, "status": "draft_review", "user_id": 7,
            "workspace_id": 70, "topic_id": 700,
        }
        with (
            patch.object(bot, "_active_session_for_msg", AsyncMock(return_value=session)),
            patch.object(db, "approve_review", AsyncMock(return_value=True)) as approve,
            patch.object(worker, "process", AsyncMock()) as process,
        ):
            await bot.on_approve(msg)
        approve.assert_awaited_once_with(9, 7, msg.chat.id)
        process.assert_not_awaited()
        self.assertIn("поставлен в очередь", msg.answer.await_args.args[0])

    async def test_worker_holds_topic_lock_across_extract_and_summary(self) -> None:
        events: list[str] = []

        @asynccontextmanager
        async def locked(_topic_id):
            events.append("lock")
            try:
                yield
            finally:
                events.append("unlock")

        async def extracted(_session_id):
            events.append("extract")

        async def summarized(_workspace_id, _topic_id):
            events.append("summary")

        async def projected(_session_id):
            events.append("project")

        async def evaluated(_session_id):
            events.append("eval")

        job = {"workspace_id": 70, "topic_id": 700, "session_id": 9}
        with (
            patch.object(db, "topic_advisory_lock", locked),
            patch.object(worker.extract, "run", extracted),
            patch.object(worker.memory, "project_session", projected),
            patch.object(worker.summary, "run", summarized),
            patch.object(worker.evaljob, "run", evaluated),
        ):
            await worker.process(job)
        self.assertEqual(
            events, ["lock", "extract", "project", "summary", "unlock", "eval"]
        )

    def test_migration_has_security_and_durable_invariants(self) -> None:
        sql = Path("migrations/006_multitenancy_review_jobs.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("UNIQUE (workspace_id, name)", sql)
        self.assertIn("WHERE status IN ('active', 'draft_review')", sql)
        self.assertIn("idempotency_key TEXT NOT NULL UNIQUE", sql)
        self.assertIn("('queued','running','retry_wait','succeeded','dead')", sql)
        self.assertIn("FOREIGN KEY (workspace_id, source_node_id)", sql)

    def test_recovery_migration_enqueues_historical_finalized_sessions(self) -> None:
        sql = Path("migrations/011_recover_finalized_jobs.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("s.status='finalized'", sql)
        self.assertIn("NOT EXISTS", sql)
        self.assertIn("ON CONFLICT (session_id) DO NOTHING", sql)


if __name__ == "__main__":
    unittest.main()
