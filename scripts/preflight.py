"""Launcher and setup preflight checks.

This module is intentionally usable from PowerShell 5.1 scripts.  It reads
``.env`` through python-dotenv, validates the runtime environment, and emits
machine-readable launcher settings so PowerShell does not have to parse env
files with regexes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


REQUIRED_PYTHON = (3, 12)
EXPECTED_EMBED_DIM = 640
DEFAULT_EMBED_PORT = 8300
DEFAULT_READY_TIMEOUT_SEC = 180

REQUIRED_IMPORTS: dict[str, str] = {
    "aiogram": "aiogram",
    "openai": "openai",
    "httpx": "httpx",
    "asyncpg": "asyncpg",
    "pgvector": "pgvector",
    "python-dotenv": "dotenv",
    "huggingface_hub": "huggingface_hub",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
}
EMBED_REQUIRED_IMPORTS: dict[str, str] = {
    "sentence-transformers": "sentence_transformers",
}

MODEL_REQUIRED_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
)


class PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fail(message: str) -> None:
    print(f"[preflight] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _warn(message: str) -> None:
    print(f"[preflight] warning: {message}", file=sys.stderr)


def _load_dotenv(root: Path) -> dict[str, str]:
    try:
        from dotenv import load_dotenv
    except Exception as exc:  # pragma: no cover - covered by runtime preflight
        raise PreflightError(
            "python-dotenv is not importable from this Python. "
            "Run: powershell -File scripts\\setup.ps1"
        ) from exc

    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return dict(os.environ)


def _as_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise PreflightError(f"{name} must be an integer, got {raw!r}") from exc


def _required(env: dict[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise PreflightError(f"missing env var: {name}")
    return value


def _is_placeholder_token(token: str) -> bool:
    lower = token.strip().lower()
    return lower in {
        "put-your-botfather-token-here",
        "your-token",
        "telegram-token",
        "changeme",
    }


def _ensure_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PreflightError(f"{name} must be an absolute http(s) URL, got {value!r}")


def _ensure_openai_v1_url(name: str, value: str) -> None:
    _ensure_url(name, value)
    if not urlparse(value).path.rstrip("/").endswith("/v1"):
        raise PreflightError(f"{name} must point to an OpenAI-compatible /v1 URL")


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    return host.lower() in {"localhost", "127.0.0.1", "::1"}


def _health_url_from_base(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def resolve_embed_config(
    env: dict[str, str] | None = None, root: Path | None = None
) -> dict[str, Any]:
    """Resolve embedding launcher settings from app-compatible environment.

    Missing EMBED_MODE is backward-compatible but conservative:
    - no EMBED_BASE_URL -> disabled
    - non-empty EMBED_BASE_URL -> external

    This prevents the launcher from starting a bundled server just because a
    user configured an external endpoint.
    """

    root = root or project_root()
    env = dict(env or os.environ)
    warnings: list[str] = []

    raw_mode = env.get("EMBED_MODE", "").strip().lower()
    raw_base_url = env.get("EMBED_BASE_URL", "").strip()
    port = _as_int(env, "EMBED_PORT", DEFAULT_EMBED_PORT)
    timeout = _as_int(env, "EMBED_READY_TIMEOUT_SEC", DEFAULT_READY_TIMEOUT_SEC)
    device = env.get("EMBED_DEVICE", "auto").strip().lower() or "auto"
    model_dir = env.get("EMBED_MODEL_DIR", "models/harrier-oss-v1-270m").strip()

    if device not in {"auto", "cpu", "cuda"}:
        raise PreflightError("EMBED_DEVICE must be one of: auto, cpu, cuda")
    if timeout < 10:
        raise PreflightError("EMBED_READY_TIMEOUT_SEC must be at least 10")

    if raw_mode:
        mode = raw_mode
    elif raw_base_url:
        mode = "external"
        warnings.append(
            "EMBED_MODE is unset; inferred external because EMBED_BASE_URL is set. "
            "Set EMBED_MODE=bundled if run.ps1 should start the local server."
        )
    else:
        mode = "disabled"

    if mode not in {"disabled", "bundled", "external"}:
        raise PreflightError("EMBED_MODE must be one of: disabled, bundled, external")

    base_url = raw_base_url
    start_local = False
    health_url = ""
    host = "127.0.0.1"

    if mode == "disabled":
        base_url = ""
        if raw_base_url:
            warnings.append("EMBED_BASE_URL is ignored because EMBED_MODE=disabled")
    elif mode == "bundled":
        start_local = True
        if not base_url:
            base_url = f"http://127.0.0.1:{port}/v1"
        _ensure_openai_v1_url("EMBED_BASE_URL", base_url)
        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        if parsed.port is not None:
            port = parsed.port
        if not _is_loopback_host(host):
            raise PreflightError(
                "EMBED_MODE=bundled requires a loopback EMBED_BASE_URL "
                "(localhost or 127.0.0.1)"
            )
        health_url = _health_url_from_base(base_url)
    else:
        if not base_url:
            raise PreflightError("EMBED_MODE=external requires EMBED_BASE_URL")
        _ensure_openai_v1_url("EMBED_BASE_URL", base_url)

    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = root / model_path

    return {
        "embed_mode": mode,
        "embed_enabled": mode != "disabled",
        "start_local_embed": start_local,
        "embed_base_url": base_url,
        "embed_host": host,
        "embed_port": port,
        "embed_health_url": health_url,
        "embed_expected_dim": EXPECTED_EMBED_DIM,
        "embed_model_dir": str(model_path),
        "embed_device": device,
        "embed_ready_timeout_sec": timeout,
        "bot_restart_initial_delay_sec": _as_int(env, "BOT_RESTART_INITIAL_DELAY_SEC", 5),
        "bot_restart_max_delay_sec": _as_int(env, "BOT_RESTART_MAX_DELAY_SEC", 60),
        "bot_restart_max_fast_failures": _as_int(env, "BOT_RESTART_MAX_FAST_FAILURES", 3),
        "warnings": warnings,
    }


def model_status(model_dir: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for rel in MODEL_REQUIRED_FILES:
        path = model_dir / rel
        if not path.is_file():
            missing.append(rel)
    weights = model_dir / "model.safetensors"
    if weights.exists() and weights.stat().st_size < 10_000_000:
        missing.append("model.safetensors (too small; likely partial download)")
    return not missing, missing


def check_python_version() -> None:
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise PreflightError(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} is required; "
            f"current is {sys.version.split()[0]}"
        )


def check_imports(imports: dict[str, str], label: str, hint: str) -> None:
    missing: list[str] = []
    for package, import_name in imports.items():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            missing.append(f"{package} ({exc.__class__.__name__}: {exc})")
    if missing:
        joined = "\n  - ".join(missing)
        raise PreflightError(
            f"venv is missing {label} dependencies:\n"
            f"  - {joined}\n"
            f"Run: {hint}"
        )


def check_runtime_imports() -> None:
    check_imports(
        REQUIRED_IMPORTS,
        "runtime",
        "powershell -File scripts\\setup.ps1",
    )


def check_embedding_imports() -> None:
    check_imports(
        EMBED_REQUIRED_IMPORTS,
        "bundled embedding",
        "powershell -File scripts\\setup.ps1 -WithEmbeddings",
    )


def check_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except Exception as exc:
        if device == "cuda":
            raise PreflightError(
                "EMBED_DEVICE=cuda but torch is not importable"
            ) from exc
        return "cpu"

    has_cuda = bool(torch.cuda.is_available())
    if device == "cuda" and not has_cuda:
        raise PreflightError(
            "EMBED_DEVICE=cuda but CUDA is unavailable. "
            "Set EMBED_DEVICE=auto or EMBED_DEVICE=cpu."
        )
    if device == "auto":
        return "cuda" if has_cuda else "cpu"
    return device


def check_app_config(root: Path, docker: bool = False) -> dict[str, Any]:
    env = _load_dotenv(root)
    token = _required(env, "TELEGRAM_BOT_TOKEN")
    if _is_placeholder_token(token):
        raise PreflightError(
            "TELEGRAM_BOT_TOKEN still has the template placeholder value"
        )
    _required(env, "DATABASE_URL")
    openai_base_url = env.get("OPENAI_BASE_URL", "")
    _ensure_openai_v1_url("OPENAI_BASE_URL", openai_base_url)
    if docker and _is_loopback_host(urlparse(openai_base_url).hostname):
        raise PreflightError(
            "docker-compose cannot use a loopback OPENAI_BASE_URL because "
            "localhost points at the bot container. Use "
            "OPENAI_BASE_URL=http://host.docker.internal:20128/v1 or a "
            "network-reachable OpenAI-compatible endpoint."
        )
    for name in ("DIALOG_MODEL", "EXTRACT_MODEL", "SUMMARY_MODEL", "EVAL_MODEL"):
        _required(env, name)

    cfg = resolve_embed_config(env, root)
    if docker and cfg["embed_mode"] == "bundled":
        raise PreflightError(
            "EMBED_MODE=bundled is supported only by local scripts\\run.ps1. "
            "For docker-compose use EMBED_MODE=disabled or EMBED_MODE=external "
            "with EMBED_BASE_URL=http://host.docker.internal:8300/v1."
        )
    if docker and cfg["embed_mode"] == "external":
        host = urlparse(cfg["embed_base_url"]).hostname
        if _is_loopback_host(host):
            raise PreflightError(
                "EMBED_MODE=external in docker-compose cannot use a loopback "
                "EMBED_BASE_URL because localhost points at the bot container. "
                "Use EMBED_BASE_URL=http://host.docker.internal:8300/v1 or a "
                "network-reachable embedding endpoint."
            )

    if cfg["embed_mode"] == "bundled":
        check_embedding_imports()
        ok, missing = model_status(Path(cfg["embed_model_dir"]))
        if not ok:
            raise PreflightError(
                "bundled embedding model is incomplete. Missing: "
                + ", ".join(missing)
                + ". Run: powershell -File scripts\\setup.ps1 -WithEmbeddings"
            )
        resolved = check_device(cfg["embed_device"])
        if cfg["embed_device"] == "auto" and resolved == "cpu":
            _warn("CUDA unavailable; bundled embeddings will run on CPU")
    return cfg


def check_port_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
    if result == 0:
        raise PreflightError(
            f"{host}:{port} is already in use. If this is an external embed "
            "server, set EMBED_MODE=external; otherwise stop that process."
        )


def wait_health(url: str, timeout_sec: int, expected_dim: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok" and int(payload.get("dim", -1)) == expected_dim:
                return payload
            last_error = f"bad health payload: {payload!r}"
        except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise PreflightError(f"health check timed out for {url}: {last_error}")


def cmd_runtime(_: argparse.Namespace) -> None:
    check_python_version()
    check_runtime_imports()
    print(f"[preflight] runtime ok: Python {sys.version.split()[0]}")


def cmd_config(args: argparse.Namespace) -> None:
    cfg = check_app_config(Path(args.root).resolve(), docker=args.docker)
    for warning in cfg["warnings"]:
        _warn(warning)
    print(f"[preflight] config ok: EMBED_MODE={cfg['embed_mode']}")


def cmd_launcher_env(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    env = _load_dotenv(root)
    cfg = resolve_embed_config(env, root)
    if args.json:
        print(json.dumps(cfg, ensure_ascii=True))
    else:
        for key, value in cfg.items():
            print(f"{key}={value}")


def cmd_model(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    env = _load_dotenv(root)
    cfg = resolve_embed_config(env, root)
    model_dir = Path(args.model_dir or cfg["embed_model_dir"])
    ok, missing = model_status(model_dir)
    if not ok:
        raise PreflightError(
            "embedding model is incomplete. Missing: " + ", ".join(missing)
        )
    print(f"[preflight] model ok: {model_dir}")


def cmd_port_free(args: argparse.Namespace) -> None:
    check_port_free(args.host, int(args.port))
    print(f"[preflight] port free: {args.host}:{args.port}")


def cmd_health(args: argparse.Namespace) -> None:
    payload = wait_health(args.url, args.timeout, args.expected_dim)
    print(json.dumps(payload, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(project_root()))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("runtime")
    p.set_defaults(func=cmd_runtime)

    p = sub.add_parser("config")
    p.add_argument("--docker", action="store_true")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("launcher-env")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_launcher_env)

    p = sub.add_parser("model")
    p.add_argument("--model-dir", default="")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("port-free")
    p.add_argument("--host", required=True)
    p.add_argument("--port", required=True, type=int)
    p.set_defaults(func=cmd_port_free)

    p = sub.add_parser("health")
    p.add_argument("--url", required=True)
    p.add_argument("--timeout", type=int, default=DEFAULT_READY_TIMEOUT_SEC)
    p.add_argument("--expected-dim", type=int, default=EXPECTED_EMBED_DIM)
    p.set_defaults(func=cmd_health)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except PreflightError as exc:
        print(f"[preflight] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
