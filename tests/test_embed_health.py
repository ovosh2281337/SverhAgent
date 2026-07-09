import unittest
from types import SimpleNamespace

from src import config, embed


class _FakeEmbeddings:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    async def create(self, **_kwargs):
        if self.error:
            raise self.error
        return self.result


class _FakeClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.embeddings = _FakeEmbeddings(result, error)


class EmbedHealthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.saved = {
            name: getattr(embed, name)
            for name in (
                "_enabled", "_client", "_status", "_attempts", "_successes",
                "_failures", "_last_error", "_last_success_at", "_last_failure_at",
            )
        }
        embed._enabled = True
        embed._status = "unknown"
        embed._attempts = embed._successes = embed._failures = 0
        embed._last_error = embed._last_success_at = embed._last_failure_at = None

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            setattr(embed, name, value)

    async def test_valid_vector_marks_backend_healthy(self) -> None:
        response = SimpleNamespace(data=[SimpleNamespace(
            embedding=[0.0] * config.EMBED_DIM
        )])
        embed._client = _FakeClient(result=response)
        vector = await embed.embed("health", required=True)
        self.assertEqual(len(vector), config.EMBED_DIM)
        self.assertEqual(embed.metrics()["status"], "healthy")
        self.assertEqual(embed.metrics()["successes"], 1)

    async def test_required_failure_raises_and_marks_degraded(self) -> None:
        embed._client = _FakeClient(error=OSError("endpoint down"))
        with self.assertRaises(embed.EmbeddingError):
            await embed.embed("fact", required=True)
        self.assertEqual(embed.metrics()["status"], "degraded")
        self.assertEqual(embed.metrics()["failures"], 1)

    async def test_optional_failure_is_observable_but_nonfatal(self) -> None:
        embed._client = _FakeClient(error=OSError("endpoint down"))
        self.assertIsNone(await embed.embed("query"))
        self.assertEqual(embed.metrics()["status"], "degraded")


if __name__ == "__main__":
    unittest.main()
