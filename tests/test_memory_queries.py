import unittest
from unittest.mock import AsyncMock, patch

from src import config, db


class _FakePool:
    def __init__(self) -> None:
        self.fetch = AsyncMock(return_value=[])
        self.fetchrow = AsyncMock(return_value=None)


class MemoryQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_reads_only_published_current_version(self) -> None:
        fake = _FakePool()
        with patch.object(db, "pool", AsyncMock(return_value=fake)):
            await db.search_canonical("topic", [0.0] * config.EMBED_DIM, 6, 9)
        sql = fake.fetch.await_args.args[0]
        self.assertIn("s.status='extracted'", sql)
        self.assertIn("e.embed_version=$5", sql)
        self.assertIn("e.grounding_version=$6", sql)
        self.assertIn("e.grounding_status='verified'", sql)
        self.assertIn("(e.embedding <=> $2) <= $7", sql)
        self.assertEqual(fake.fetch.await_args.args[5], config.EMBED_TEXT_VERSION)
        self.assertEqual(fake.fetch.await_args.args[6], config.GROUNDING_VERSION)
        self.assertEqual(fake.fetch.await_args.args[7], config.RAG_MAX_DISTANCE)

    async def test_summary_reads_only_published_current_version(self) -> None:
        fake = _FakePool()
        with patch.object(db, "pool", AsyncMock(return_value=fake)):
            await db.canonical_for_topic("topic")
        sql = fake.fetch.await_args.args[0]
        self.assertIn("s.status='extracted'", sql)
        self.assertIn("e.embed_version=$2", sql)
        self.assertIn("e.grounding_version=$3", sql)
        self.assertIn("e.grounding_status='verified'", sql)
        self.assertIn("extracted_item_provenance", sql)

    async def test_dedup_sees_current_extraction_but_not_other_partial_sessions(self) -> None:
        fake = _FakePool()
        with patch.object(db, "pool", AsyncMock(return_value=fake)):
            await db.nearest_canonical("topic", [0.0] * config.EMBED_DIM, 42)
        sql = fake.fetchrow.await_args.args[0]
        self.assertIn("s.status='extracted' OR e.session_id=$5", sql)
        self.assertIn("e.grounding_version=$4", sql)
        self.assertIn("e.grounding_status='verified'", sql)
        self.assertEqual(fake.fetchrow.await_args.args[4], config.GROUNDING_VERSION)
        self.assertEqual(fake.fetchrow.await_args.args[5], 42)


if __name__ == "__main__":
    unittest.main()
