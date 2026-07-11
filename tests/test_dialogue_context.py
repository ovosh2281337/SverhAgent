from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src import dialogue_context, llm


def _row(id_: int, role: str = "user", content: str = "text") -> dict:
    return {"id": id_, "role": role, "content": content}


class DialogueContextTests(unittest.IsolatedAsyncioTestCase):
    def test_oversized_message_is_split_without_data_loss(self):
        content = "0123456789" * 2_000
        with patch.object(dialogue_context.config, "DIALOG_CONTEXT_TOKENS", 1_000):
            batches = dialogue_context._bounded_batches([_row(7, content=content)])
        parts = [
            item["content"].split("\n", 1)[1]
            for batch in batches for item in batch
        ]
        self.assertGreater(len(batches), 1)
        self.assertEqual("".join(parts), content)

    async def test_long_history_is_compacted_and_only_recent_tail_is_returned(self):
        recent = [_row(i) for i in range(81, 101)]
        with (
            patch.object(
                dialogue_context.db, "context_compaction", AsyncMock(return_value=None)
            ),
             patch.object(
                dialogue_context.db, "uncompacted_history_size",
                AsyncMock(side_effect=[
                    {"messages": 100, "chars": 100_000},
                    {"messages": 92, "chars": 92_000},
                    {"messages": 84, "chars": 84_000},
                    {"messages": 76, "chars": 76_000},
                    {"messages": 68, "chars": 68_000},
                ]),
            ),
            patch.object(
                dialogue_context.db, "history_after",
                AsyncMock(side_effect=[
                    [_row(i) for i in range(1, 9)],
                    [_row(i) for i in range(9, 17)],
                    [_row(i) for i in range(17, 25)],
                    [_row(i) for i in range(25, 33)],
                ]),
            ),
            patch.object(
                dialogue_context.db, "recent_history_within_chars",
                AsyncMock(return_value=recent),
            ) as recent_mock,
            patch.object(
                dialogue_context.db, "upsert_context_compaction", AsyncMock()
            ) as save_mock,
            patch.object(
                dialogue_context.llm, "compact_history",
                AsyncMock(side_effect=["s1", "s2", "s3", "s4"]),
            ),
        ):
            rows, summary, backlog = await dialogue_context.build(7)

        self.assertEqual(rows, recent)
        self.assertEqual(summary, "s4")
        self.assertTrue(backlog)
        self.assertEqual(save_mock.await_count, 4)
        self.assertEqual(recent_mock.await_args.args[:2], (7, 32))
        self.assertGreater(recent_mock.await_args.args[2], 0)

    async def test_short_history_uses_raw_tail_without_compactor(self):
        rows = [_row(1), _row(2, "assistant")]
        with (
            patch.object(
                dialogue_context.db, "context_compaction", AsyncMock(return_value=None)
            ),
            patch.object(
                dialogue_context.db, "uncompacted_history_size",
                AsyncMock(side_effect=[
                    {"messages": 2, "chars": 500},
                    {"messages": 2, "chars": 500},
                ]),
            ),
            patch.object(
                dialogue_context.db, "history_after_all", AsyncMock(return_value=rows)
            ),
            patch.object(dialogue_context.llm, "compact_history", AsyncMock()) as compact,
        ):
            actual, summary, backlog = await dialogue_context.build(7)
        self.assertEqual(actual, rows)
        self.assertEqual(summary, "")
        self.assertFalse(backlog)
        compact.assert_not_awaited()


class DialogueBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_is_checked_before_completion(self):
        messages = [{"role": "system", "content": "x" * 100_000}]
        with (
            patch.object(llm, "_complete", AsyncMock()) as complete,
            patch.object(llm.config, "DIALOG_CONTEXT_TOKENS", 4096),
        ):
            with self.assertRaises(llm.DialogueContextExceeded):
                await llm.dialogue(messages, [], AsyncMock())
        complete.assert_not_awaited()

    async def test_max_tokens_is_bounded_before_completion(self):
        message = SimpleNamespace(content="ok", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(total_tokens=120),
        )
        with patch.object(llm, "_complete", AsyncMock(return_value=response)) as complete:
            text, spent = await llm.dialogue(
                [{"role": "user", "content": "hello"}], [], AsyncMock(),
            )
        self.assertEqual((text, spent), ("ok", 120))
        self.assertNotIn("max_tokens", complete.await_args.args[2])


if __name__ == "__main__":
    unittest.main()
