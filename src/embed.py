"""Embeddings via an OpenAI-compatible endpoint (harrier-oss-v1-270m).

EMBED_MODE=disabled intentionally disables vector features, even if a stray
EMBED_BASE_URL is present. In bundled/external modes failures are observable:
the module records health/error counters, logs state transitions, and mandatory
callers can request fail-fast behaviour. This distinction prevents an
unavailable backend from silently turning every extracted item into a new
canonical fact.

harrier is instruction-tuned with last-token pooling + L2 normalization, so
cosine distance is the right metric. Retrieval queries get an instruct prefix;
stored documents are embedded raw.
"""
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import httpx
from openai import AsyncOpenAI

from . import config

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Configured embedding backend could not produce a valid vector."""


_enabled = bool(config.EMBED_BASE_URL)
# trust_env=False: skip the OS/registry proxy for the local endpoint (see llm.py).
_client = (
    AsyncOpenAI(
        base_url=config.EMBED_BASE_URL,
        api_key=config.EMBED_API_KEY,
        http_client=httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(60.0)),
    )
    if _enabled
    else None
)

_status = "unknown" if _enabled else "disabled"
_attempts = 0
_successes = 0
_failures = 0
_last_error: Optional[str] = None
_last_success_at: Optional[str] = None
_last_failure_at: Optional[str] = None


def enabled() -> bool:
    return _enabled and _client is not None


def metrics() -> dict[str, object]:
    """Process-local health snapshot suitable for logs or a metrics adapter."""
    return {
        "status": _status,
        "attempts": _attempts,
        "successes": _successes,
        "failures": _failures,
        "last_error": _last_error,
        "last_success_at": _last_success_at,
        "last_failure_at": _last_failure_at,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_success() -> None:
    global _status, _successes, _last_error, _last_success_at
    recovered = _status == "degraded"
    _status = "healthy"
    _successes += 1
    _last_error = None
    _last_success_at = _now()
    if recovered:
        log.info("embedding backend recovered; metrics=%s", metrics())


def _record_failure(exc: Exception) -> None:
    global _status, _failures, _last_error, _last_failure_at
    _status = "degraded"
    _failures += 1
    _last_error = f"{type(exc).__name__}: {exc}"[:500]
    _last_failure_at = _now()
    log.error("embedding backend degraded; metrics=%s", metrics())


async def embed(
    text: str, *, query: bool = False, required: bool = False
) -> Optional[list[float]]:
    """Return one validated embedding.

    Optional online retrieval may use ``required=False`` and continue without
    RAG after a logged failure. Extraction/backfill use ``required=True`` so a
    configured but broken backend cannot silently bypass deduplication.
    """
    global _attempts
    if not enabled():
        if required:
            raise EmbeddingError("embeddings are required but EMBED_MODE=disabled")
        return None
    if not text.strip():
        if required:
            raise EmbeddingError("cannot embed empty text")
        return None

    payload = (config.EMBED_QUERY_PROMPT + text) if query else text
    _attempts += 1
    try:
        r = await _client.embeddings.create(model=config.EMBED_MODEL, input=payload)
        if not r.data:
            raise ValueError("embedding response has no data")
        vector = list(r.data[0].embedding)
        if len(vector) != config.EMBED_DIM:
            raise ValueError(
                f"embedding dimension {len(vector)} != configured {config.EMBED_DIM}"
            )
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("embedding contains a non-finite value")
    except Exception as exc:
        _record_failure(exc)
        if required:
            raise EmbeddingError(_last_error or "embedding request failed") from exc
        return None
    _record_success()
    return vector


async def health_check() -> dict[str, object]:
    """Probe the configured endpoint and return the updated health snapshot."""
    if enabled():
        await embed("embedding service health check", required=True)
    return metrics()
