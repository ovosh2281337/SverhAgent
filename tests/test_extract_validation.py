import unittest

from src.jobs.extract import _validate_item


class ExtractValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = {
            1: {"role": "assistant", "content": "Вы сушите PA-CF перед печатью?"},
            2: {
                "role": "user",
                "content": "Да, PA-CF сушу 12 часов при 70 C, иначе сопли.",
            },
            3: {
                "role": "assistant",
                "content": "Правильно понял: PA-CF сушите 12 часов при 70 C?",
            },
            4: {"role": "user", "content": "Да, именно так."},
            5: {
                "role": "assistant",
                "content": "Вы раньше сказали, что PETG не сушите.",
            },
            6: {
                "role": "user",
                "content": "Поправлю: PETG сушу только перед ответственными деталями.",
            },
            7: {
                "role": "user",
                "content": "После сушки PA-CF перестал пузыриться на сопле.",
            },
            8: {"role": "user", "content": "Нет, неверно."},
            9: {"role": "user", "content": "да, да, сушу"},
            10: {"role": "user", "content": "Ну да, именно."},
        }

    def _fact(
        self, *, mode="direct_assertion", origin="expert_claim",
        provenance=None, payload=None,
    ):
        return {
            "type": "fact",
            "origin": origin,
            "support_mode": mode,
            "payload": payload or {
                "statement": "PA-CF сушат 12 часов при 70 C",
                "qualifiers": "перед печатью",
            },
            "provenance": provenance or [
                {
                    "source_ref": 2,
                    "kind": "user_support",
                    "quote": "PA-CF сушу 12 часов при 70 C",
                }
            ],
            "retrieval_question": "Как эксперт сушит PA-CF?",
            "contradicts_self": False,
        }

    def test_accepts_direct_assertion_with_exact_unique_user_span(self) -> None:
        valid, error = _validate_item(self._fact(), self.messages)
        self.assertIsNone(error)
        self.assertEqual(valid.origin, "expert_claim")
        self.assertEqual(valid.support_mode, "direct_assertion")
        self.assertEqual(valid.primary_support.message_id, 2)
        self.assertEqual(valid.primary_support.kind, "user_support")

    def test_rejects_unknown_origin_instead_of_promoting_it(self) -> None:
        valid, error = _validate_item(
            self._fact(origin="model_guess"), self.messages
        )
        self.assertIsNone(valid)
        self.assertEqual(error, "unknown origin")

    def test_rejects_legacy_top_level_source_ref_shape(self) -> None:
        item = {
            "type": "fact",
            "origin": "expert_claim",
            "payload": {"statement": "PA-CF сушат 12 часов при 70 C"},
            "quote": "PA-CF сушу 12 часов при 70 C",
            "source_ref": 2,
        }
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(valid)
        self.assertEqual(error, "unknown support_mode")

    def test_rejects_boolean_source_ref(self) -> None:
        item = self._fact(provenance=[
            {
                "source_ref": True,
                "kind": "user_support",
                "quote": "PA-CF сушу 12 часов при 70 C",
            }
        ])
        valid, error = _validate_item(item, {True: self.messages[2]})
        self.assertIsNone(valid)
        self.assertIn("source_ref", error)

    def test_rejects_agent_message_as_user_support(self) -> None:
        item = self._fact(provenance=[
            {
                "source_ref": 1,
                "kind": "user_support",
                "quote": "Вы сушите PA-CF",
            }
        ])
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(valid)
        self.assertEqual(error, "user_support must reference an expert message")

    def test_rejects_non_unique_quote(self) -> None:
        item = self._fact(provenance=[
            {"source_ref": 9, "kind": "user_support", "quote": "да"}
        ])
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(valid)
        self.assertIn("missing or non-unique", error)

    def test_accepts_contextual_answer_only_with_prior_question_context(self) -> None:
        item = self._fact(
            mode="contextual_answer",
            provenance=[
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
            ],
        )
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(error)
        self.assertEqual(valid.support_mode, "contextual_answer")

    def test_rejects_contextual_answer_when_context_after_support(self) -> None:
        item = self._fact(
            mode="contextual_answer",
            provenance=[
                {
                    "source_ref": 3,
                    "kind": "question_context",
                    "quote": "Правильно понял",
                },
                {
                    "source_ref": 2,
                    "kind": "user_support",
                    "quote": "PA-CF сушу 12 часов при 70 C",
                },
            ],
        )
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(valid)
        self.assertEqual(error, "question_context must precede user_support")

    def test_accepts_explicit_confirmation_as_confirmed_hypothesis(self) -> None:
        item = self._fact(
            mode="explicit_confirmation",
            origin="confirmed_hypothesis",
            provenance=[
                {
                    "source_ref": 3,
                    "kind": "hypothesis_target",
                    "quote": "PA-CF сушите 12 часов при 70 C?",
                },
                {"source_ref": 4, "kind": "confirmation", "quote": "Да, именно так."},
            ],
        )
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(error)
        self.assertEqual(valid.primary_support.kind, "confirmation")

    def test_rejects_negative_or_ambiguous_confirmation(self) -> None:
        negative = self._fact(
            mode="explicit_confirmation",
            origin="confirmed_hypothesis",
            provenance=[
                {
                    "source_ref": 3,
                    "kind": "hypothesis_target",
                    "quote": "PA-CF сушите 12 часов при 70 C?",
                },
                {"source_ref": 8, "kind": "confirmation", "quote": "Нет, неверно."},
            ],
        )
        self.assertEqual(
            _validate_item(negative, self.messages)[1],
            "negative answer cannot be confirmation",
        )

        ambiguous = self._fact(
            mode="explicit_confirmation",
            origin="confirmed_hypothesis",
            provenance=[
                {
                    "source_ref": 3,
                    "kind": "hypothesis_target",
                    "quote": "PA-CF сушите 12 часов при 70 C?",
                },
                {"source_ref": 10, "kind": "confirmation", "quote": "Ну да"},
            ],
        )
        self.assertEqual(
            _validate_item(ambiguous, self.messages)[1],
            "implicit confirmation is ambiguous",
        )

    def test_accepts_correction_with_target_and_later_user_support(self) -> None:
        item = self._fact(
            mode="correction",
            payload={
                "statement": "PETG сушат перед ответственными деталями",
                "qualifiers": "только для ответственных деталей",
            },
            provenance=[
                {
                    "source_ref": 5,
                    "kind": "correction_target",
                    "quote": "PETG не сушите",
                },
                {
                    "source_ref": 6,
                    "kind": "user_support",
                    "quote": "PETG сушу только перед ответственными деталями",
                },
            ],
        )
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(error)
        self.assertEqual(valid.support_mode, "correction")

    def test_multi_turn_synthesis_requires_two_expert_messages(self) -> None:
        ok = self._fact(
            mode="multi_turn_synthesis",
            provenance=[
                {
                    "source_ref": 2,
                    "kind": "user_support",
                    "quote": "PA-CF сушу 12 часов при 70 C",
                },
                {
                    "source_ref": 7,
                    "kind": "user_support",
                    "quote": "PA-CF перестал пузыриться",
                },
            ],
        )
        valid, error = _validate_item(ok, self.messages)
        self.assertIsNone(error)
        self.assertEqual(valid.support_mode, "multi_turn_synthesis")

        bad = self._fact(mode="multi_turn_synthesis")
        self.assertEqual(
            _validate_item(bad, self.messages)[1],
            "multi_turn_synthesis requires two user messages",
        )

    def test_rejects_invalid_payload_schema(self) -> None:
        item = self._fact()
        item["payload"] = {"statement": "", "confidence": 0.9}
        valid, error = _validate_item(item, self.messages)
        self.assertIsNone(valid)
        self.assertIn("payload", error)


if __name__ == "__main__":
    unittest.main()
