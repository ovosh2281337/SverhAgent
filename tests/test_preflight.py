import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import preflight


class PreflightEnvTests(unittest.TestCase):
    def test_bundled_defaults_to_loopback_v1_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = preflight.resolve_embed_config(
                {"EMBED_MODE": "bundled", "EMBED_PORT": "8311"},
                Path(tmp),
            )
        self.assertTrue(cfg["start_local_embed"])
        self.assertEqual(cfg["embed_base_url"], "http://127.0.0.1:8311/v1")
        self.assertEqual(cfg["embed_health_url"], "http://127.0.0.1:8311/health")

    def test_external_does_not_start_local_process(self) -> None:
        cfg = preflight.resolve_embed_config(
            {
                "EMBED_MODE": "external",
                "EMBED_BASE_URL": "https://embed.example/v1",
            }
        )
        self.assertEqual(cfg["embed_mode"], "external")
        self.assertFalse(cfg["start_local_embed"])

    def test_missing_mode_with_nonempty_base_is_external_not_bundled(self) -> None:
        cfg = preflight.resolve_embed_config(
            {"EMBED_BASE_URL": "http://127.0.0.1:8300/v1"}
        )
        self.assertEqual(cfg["embed_mode"], "external")
        self.assertFalse(cfg["start_local_embed"])
        self.assertTrue(cfg["warnings"])

    def test_disabled_ignores_stray_base_url(self) -> None:
        cfg = preflight.resolve_embed_config(
            {
                "EMBED_MODE": "disabled",
                "EMBED_BASE_URL": "https://embed.example/v1",
            }
        )
        self.assertEqual(cfg["embed_base_url"], "")
        self.assertFalse(cfg["embed_enabled"])
        self.assertTrue(cfg["warnings"])

    def test_dotenv_parsing_matches_application_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                'EMBED_BASE_URL = http://127.0.0.1:8300/v1\n'
                'EMBED_BASE_URL=""\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                env = preflight._load_dotenv(root)
        self.assertEqual(env.get("EMBED_BASE_URL"), "")
        cfg = preflight.resolve_embed_config(env, root)
        self.assertEqual(cfg["embed_mode"], "disabled")

    def test_bundled_rejects_non_loopback_url(self) -> None:
        with self.assertRaises(preflight.PreflightError):
            preflight.resolve_embed_config(
                {
                    "EMBED_MODE": "bundled",
                    "EMBED_BASE_URL": "https://embed.example/v1",
                }
            )

    def test_openai_base_url_must_end_with_v1(self) -> None:
        with self.assertRaises(preflight.PreflightError):
            preflight.resolve_embed_config(
                {
                    "EMBED_MODE": "external",
                    "EMBED_BASE_URL": "https://embed.example",
                }
            )

    def test_partial_model_directory_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}", encoding="utf-8")
            ok, missing = preflight.model_status(model)
        self.assertFalse(ok)
        self.assertIn("model.safetensors", " ".join(missing))

    def test_runtime_imports_do_not_require_embedding_extra(self) -> None:
        def fake_import(name: str):
            if name == "sentence_transformers":
                raise ModuleNotFoundError(name)
            return object()

        with patch.object(preflight.importlib, "import_module", side_effect=fake_import):
            preflight.check_runtime_imports()

    def test_bundled_config_requires_embedding_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=123456:valid-looking-token",
                        "DATABASE_URL=postgresql://kb:kb@localhost:5432/kb",
                        "OPENAI_BASE_URL=http://localhost:8000/v1",
                        "DIALOG_MODEL=chat",
                        "EXTRACT_MODEL=chat",
                        "SUMMARY_MODEL=chat",
                        "EVAL_MODEL=chat",
                        "GROUND_MODEL=chat",
                        "EMBED_MODE=bundled",
                    ]
                ),
                encoding="utf-8",
            )

            def fake_import(name: str):
                if name == "sentence_transformers":
                    raise ModuleNotFoundError(name)
                return object()

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(preflight.importlib, "import_module", side_effect=fake_import):
                    with self.assertRaisesRegex(preflight.PreflightError, "WithEmbeddings"):
                        preflight.check_app_config(root)

    def test_docker_rejects_external_loopback_embedding_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=123456:valid-looking-token",
                        "DATABASE_URL=postgresql://kb:kb@db:5432/kb",
                        "OPENAI_BASE_URL=http://host.docker.internal:8000/v1",
                        "DIALOG_MODEL=chat",
                        "EXTRACT_MODEL=chat",
                        "SUMMARY_MODEL=chat",
                        "EVAL_MODEL=chat",
                        "GROUND_MODEL=chat",
                        "EMBED_MODE=external",
                        "EMBED_BASE_URL=http://localhost:8300/v1",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(preflight.PreflightError):
                    preflight.check_app_config(root, docker=True)


if __name__ == "__main__":
    unittest.main()
