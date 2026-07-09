import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src import agent, bot, db


class _FakeMsg:
    def __init__(self, user_id: int = 42) -> None:
        self.from_user = SimpleNamespace(
            id=user_id,
            username="expert",
            full_name="Expert User",
        )
        self.chat = SimpleNamespace(id=1001)
        self.bot = SimpleNamespace(send_chat_action=AsyncMock())
        self.answer = AsyncMock()


def _session(session_id: int = 77, *, status: str = "active", user_id: int = 42):
    return {
        "id": session_id,
        "topic": "topic",
        "status": status,
        "telegram_user_id": user_id,
    }


class LifecycleLockTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        bot._locks.clear()

    def tearDown(self) -> None:
        bot._locks.clear()

    async def test_finish_waits_for_inflight_turn_lock(self) -> None:
        msg = _FakeMsg()
        sess = _session(777)
        lock = bot._lock(sess["id"])
        await lock.acquire()

        async def _noop_postprocess(*_args, **_kwargs):
            return None

        with (
            patch.object(bot, "_active_session_for_msg", AsyncMock(return_value=sess)),
            patch.object(db, "get_session", AsyncMock(return_value=sess)),
            patch.object(db, "finish_session", AsyncMock()) as finish_session,
            patch.object(bot, "_postprocess", _noop_postprocess),
        ):
            task = asyncio.create_task(bot.on_finish(msg))
            await asyncio.sleep(0.05)
            finish_session.assert_not_called()

            lock.release()
            await asyncio.wait_for(task, timeout=1)

        finish_session.assert_awaited_once_with(sess["id"])
        self.assertIn(sess["id"], bot._locks)
        self.assertTrue(any("Сессия завершена" in c.args[0] for c in msg.answer.await_args_list))

    async def test_stale_expert_turn_after_finish_or_reset_does_not_write_or_run_llm(self) -> None:
        msg = _FakeMsg()
        stale_sess = _session(778)
        lock = bot._lock(stale_sess["id"])
        await lock.acquire()

        with (
            patch.object(db, "get_session", AsyncMock(return_value=None)),
            patch.object(db, "add_message", AsyncMock()) as add_message,
            patch.object(agent, "run_turn", AsyncMock()) as run_turn,
        ):
            task = asyncio.create_task(bot._expert_turn(msg, stale_sess, "late message"))
            await asyncio.sleep(0.05)
            add_message.assert_not_called()

            lock.release()
            await asyncio.wait_for(task, timeout=1)

        add_message.assert_not_called()
        run_turn.assert_not_called()
        msg.bot.send_chat_action.assert_not_called()
        self.assertTrue(any("Сессия уже завершена" in c.args[0] for c in msg.answer.await_args_list))

    async def test_reset_revalidates_status_after_acquiring_lock(self) -> None:
        msg = _FakeMsg()
        initial = _session(779, status="active")
        fresh_finished = _session(779, status="finished")

        with (
            patch.object(bot, "_active_session_for_msg", AsyncMock(return_value=initial)),
            patch.object(db, "get_session", AsyncMock(return_value=fresh_finished)),
            patch.object(db, "delete_session", AsyncMock()) as delete_session,
        ):
            await bot.on_reset(msg)

        delete_session.assert_not_called()
        self.assertTrue(any("Активной сессии нет" in c.args[0] for c in msg.answer.await_args_list))


if __name__ == "__main__":
    unittest.main()
