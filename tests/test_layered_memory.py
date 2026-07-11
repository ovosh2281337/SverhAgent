import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src import memory, retrieval
from src.jobs import hierarchy


class _Pool:
    def __init__(self, *, allowed=True):
        self.fetchval = AsyncMock(return_value=allowed)
        self.fetch = AsyncMock(return_value=[])


class LayeredMemoryTests(unittest.IsolatedAsyncioTestCase):
    def test_claim_statement_keeps_qualifiers(self) -> None:
        self.assertEqual(
            memory.statement_from_payload(
                {"statement": "Порог 42", "qualifiers": "только ночью"}
            ),
            "Порог 42 (только ночью)",
        )

    def test_hierarchy_chunks_do_not_drop_values(self) -> None:
        self.assertEqual(
            list(hierarchy._chunks([1, 2, 3, 4, 5], 2)),
            [[1, 2], [3, 4], [5]],
        )

    def test_retrieval_router_covers_special_memory_tasks(self) -> None:
        self.assertEqual(retrieval.route_query("Кто говорил про SLA?"), "provenance")
        self.assertEqual(
            retrieval.route_query("Какие тут противоречия?"), "contradictions"
        )
        self.assertEqual(retrieval.route_query("Что изменилось раньше?"), "temporal")
        self.assertEqual(retrieval.route_query("Прямой факт"), "fact")

    async def test_retrieval_rejects_non_member(self) -> None:
        fake = _Pool(allowed=False)
        with patch.object(retrieval.db, "pool", AsyncMock(return_value=fake)):
            with self.assertRaises(PermissionError):
                await retrieval.retrieve_context(1, 2, 3, "secret")

    async def test_fts_only_query_types_vector_parameters(self) -> None:
        fake = _Pool()
        with patch.object(retrieval.db, "pool", AsyncMock(return_value=fake)):
            self.assertEqual(await retrieval._hybrid(1, 2, "42", None, limit=8), [])
        sql = fake.fetch.await_args.args[0]
        self.assertIn("$3::vector", sql)
        self.assertIn("$5::text", sql)
        self.assertIn("websearch_to_tsquery('simple',$2)", sql)
        self.assertIn("workspace_id=$1", sql)

    async def test_relation_recall_displaces_lowest_flat_seed_at_limit(self) -> None:
        def seed(id_: int) -> dict:
            return {
                "claim_id": id_, "topic_id": 2, "type": "fact",
                "origin": "expert_claim", "payload": {"statement": str(id_)},
                "normalized_statement": str(id_), "id": id_, "quote": str(id_),
                "support_mode": "direct_assertion", "score": 1.0 / id_,
                "user_ids": [3], "expert_names": "expert",
                "confirmation_count": 1, "relation_type": None,
            }
        linked = {
            **seed(99), "relation_type": "contradicts",
        }
        fake = _Pool()
        fake.fetch.side_effect = [[seed(1), seed(2)], [linked]]
        with patch.object(retrieval.db, "pool", AsyncMock(return_value=fake)):
            rows = await retrieval._hybrid(
                1, 2, "unique", None, limit=2, route="contradictions"
            )
        self.assertEqual([row["claim_id"] for row in rows], [1, 99])
        self.assertEqual(rows[1]["relation_type"], "contradicts")

    def test_migrations_enforce_layered_memory_boundaries(self) -> None:
        layered = Path("migrations/007_layered_memory.sql").read_text("utf-8")
        public = Path("migrations/008_public_collection.sql").read_text("utf-8")
        hierarchy_sql = Path(
            "migrations/009_hierarchical_summaries.sql"
        ).read_text("utf-8")
        self.assertIn("extracted_items_tenant_duplicate_fkey", layered)
        self.assertIn("count(DISTINCT evidence.user_id)", layered)
        self.assertIn("memory_claim_evidence", layered)
        self.assertIn("sessions_workspace_member_fkey", public)
        self.assertIn("u.telegram_user_id IS NOT NULL", public)
        self.assertIn("memory_summary_claims", hierarchy_sql)


if __name__ == "__main__":
    unittest.main()
