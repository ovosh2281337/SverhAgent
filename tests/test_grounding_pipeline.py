import unittest
from unittest.mock import AsyncMock, patch

from src.jobs import extract


_VERIFIED = {
    "verdict": "verified",
    "reason": "supported by expert evidence",
    "unsupported_atoms": [],
    "ambiguous": False,
}

_PARTIAL = {
    "verdict": "partial",
    "reason": "temperature is supported, duration is not",
    "unsupported_atoms": ["12 часов"],
    "ambiguous": False,
}


def _messages():
    return {
        1: {"role": "assistant", "content": "Вы сушите PA-CF перед печатью?"},
        2: {
            "role": "user",
            "content": "Да, PA-CF сушу 12 часов при 70 C, иначе сопли.",
        },
        3: {"role": "user", "content": "PETG иногда сушу перед важными деталями."},
    }


def _direct_candidate():
    return {
        "type": "fact",
        "origin": "expert_claim",
        "support_mode": "direct_assertion",
        "payload": {
            "statement": "PA-CF сушат 12 часов при 70 C",
            "qualifiers": "перед печатью",
        },
        "provenance": [
            {
                "source_ref": 2,
                "kind": "user_support",
                "quote": "PA-CF сушу 12 часов при 70 C",
            }
        ],
    }


def _contextual_candidate():
    item = _direct_candidate()
    item["support_mode"] = "contextual_answer"
    item["provenance"] = [
        {
            "source_ref": 1,
            "kind": "question_context",
            "quote": "Вы сушите PA-CF перед печатью?",
        },
        {
            "source_ref": 2,
            "kind": "user_support",
            "quote": "Да, PA-CF сушу 12 часов при 70 C",
        },
    ]
    return item


class GroundingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_item_is_embedded_and_saved_with_provenance(self) -> None:
        seen: set[str] = set()
        with (
            patch.object(extract.embed, "enabled", return_value=True),
            patch.object(
                extract.embed, "embed", AsyncMock(return_value=[0.0, 0.1])
            ) as embed_mock,
            patch.object(
                extract.db, "nearest_canonical", AsyncMock(return_value=None)
            ),
            patch.object(
                extract.db, "add_extracted_item", AsyncMock(return_value=123)
            ) as add_mock,
            patch.object(
                extract.db, "add_extraction_rejection", AsyncMock()
            ) as reject_mock,
            patch.object(
                extract.llm, "ground_extraction", AsyncMock(return_value=_VERIFIED)
            ) as ground_mock,
            patch.object(extract.llm, "repair_extraction", AsyncMock()) as repair_mock,
        ):
            n = await extract._process_raw_candidate(
                10, "topic", _contextual_candidate(), "chunk", _messages(), seen, []
            )

        self.assertEqual(n, 1)
        self.assertEqual(len(seen), 1)
        ground_mock.assert_awaited_once()
        repair_mock.assert_not_awaited()
        reject_mock.assert_not_awaited()
        add_kwargs = add_mock.await_args.kwargs
        self.assertEqual(add_kwargs["grounding_status"], "verified")
        self.assertEqual(add_kwargs["support_mode"], "contextual_answer")
        self.assertEqual(add_kwargs["primary"].kind, "user_support")
        self.assertEqual(len(add_kwargs["provenance"]), 2)
        embed_text = embed_mock.await_args.args[0]
        self.assertIn("PA-CF сушу", embed_text)
        self.assertNotIn("Вы сушите PA-CF", embed_text)

    async def test_partial_item_is_saved_for_review_without_embedding(self) -> None:
        with (
            patch.object(extract.embed, "enabled", return_value=True),
            patch.object(extract.embed, "embed", AsyncMock()) as embed_mock,
            patch.object(
                extract.db, "nearest_canonical", AsyncMock()
            ) as nearest_mock,
            patch.object(
                extract.db, "add_extracted_item", AsyncMock(return_value=123)
            ) as add_mock,
            patch.object(
                extract.db, "add_extraction_rejection", AsyncMock()
            ) as reject_mock,
            patch.object(
                extract.llm, "ground_extraction", AsyncMock(return_value=_PARTIAL)
            ),
            patch.object(
                extract.llm, "repair_extraction", AsyncMock(return_value=[])
            ) as repair_mock,
        ):
            n = await extract._process_raw_candidate(
                10, "topic", _direct_candidate(), "chunk", _messages(), set(), []
            )

        self.assertEqual(n, 0)
        embed_mock.assert_not_awaited()
        nearest_mock.assert_not_awaited()
        repair_mock.assert_awaited_once()
        reject_mock.assert_not_awaited()
        self.assertEqual(add_mock.await_args.kwargs["grounding_status"], "partial")

    async def test_invalid_candidate_can_be_repaired_once(self) -> None:
        invalid = {
            "type": "fact",
            "origin": "expert_claim",
            "payload": {"statement": "PA-CF сушат 12 часов"},
            "source_ref": 2,
            "quote": "PA-CF сушу 12 часов",
        }
        with (
            patch.object(extract.embed, "enabled", return_value=True),
            patch.object(
                extract.embed, "embed", AsyncMock(return_value=[0.0, 0.1])
            ),
            patch.object(
                extract.db, "nearest_canonical", AsyncMock(return_value=None)
            ),
            patch.object(
                extract.db, "add_extracted_item", AsyncMock(return_value=123)
            ) as add_mock,
            patch.object(
                extract.db, "add_extraction_rejection", AsyncMock()
            ) as reject_mock,
            patch.object(
                extract.llm, "ground_extraction", AsyncMock(return_value=_VERIFIED)
            ) as ground_mock,
            patch.object(
                extract.llm,
                "repair_extraction",
                AsyncMock(return_value=[_direct_candidate()]),
            ) as repair_mock,
        ):
            n = await extract._process_raw_candidate(
                10, "topic", invalid, "chunk", _messages(), set(), []
            )

        self.assertEqual(n, 1)
        repair_mock.assert_awaited_once()
        ground_mock.assert_awaited_once()
        reject_mock.assert_not_awaited()
        self.assertEqual(add_mock.await_args.kwargs["grounding_status"], "verified")

    async def test_invalid_candidate_with_no_repair_is_rejected_observably(self) -> None:
        invalid = {"type": "fact", "origin": "expert_claim"}
        with (
            patch.object(
                extract.db, "add_extracted_item", AsyncMock()
            ) as add_mock,
            patch.object(
                extract.db, "add_extraction_rejection", AsyncMock(return_value=1)
            ) as reject_mock,
            patch.object(
                extract.llm, "ground_extraction", AsyncMock()
            ) as ground_mock,
            patch.object(
                extract.llm, "repair_extraction", AsyncMock(return_value=[])
            ) as repair_mock,
        ):
            n = await extract._process_raw_candidate(
                10, "topic", invalid, "chunk", _messages(), set(), []
            )

        self.assertEqual(n, 0)
        ground_mock.assert_not_awaited()
        repair_mock.assert_awaited_once()
        add_mock.assert_not_awaited()
        reject_mock.assert_awaited_once()
        self.assertEqual(reject_mock.await_args.args[1], "validation")
        self.assertEqual(reject_mock.await_args.args[3], invalid)
        self.assertTrue(reject_mock.await_args.args[4])

    async def test_overlap_duplicate_fingerprint_is_not_recounted(self) -> None:
        seen: set[str] = set()
        with (
            patch.object(extract.embed, "enabled", return_value=False),
            patch.object(extract.embed, "embed", AsyncMock(return_value=None)),
            patch.object(
                extract.db, "nearest_canonical", AsyncMock(return_value=None)
            ),
            patch.object(
                extract.db, "add_extracted_item", AsyncMock(return_value=123)
            ) as add_mock,
            patch.object(
                extract.llm, "ground_extraction", AsyncMock(return_value=_VERIFIED)
            ) as ground_mock,
            patch.object(extract.llm, "repair_extraction", AsyncMock()),
        ):
            first = await extract._process_raw_candidate(
                10, "topic", _direct_candidate(), "chunk", _messages(), seen, []
            )
            second = await extract._process_raw_candidate(
                10, "topic", _direct_candidate(), "chunk", _messages(), seen, []
            )

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(add_mock.await_count, 1)
        self.assertEqual(ground_mock.await_count, 1)


if __name__ == "__main__":
    unittest.main()
