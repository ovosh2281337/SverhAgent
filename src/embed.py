"""Embeddings via a local OpenAI-compatible endpoint (harrier-oss-v1-270m).

Optional. Without EMBED_BASE_URL every function is a no-op returning None: the
collection loop still runs, dedup treats each item as canonical, and
search_knowledge returns nothing. Serve harrier with text-embeddings-inference
(exposes /v1/embeddings) and set EMBED_BASE_URL to light up dedup + retrieval.

harrier is instruction-tuned with last-token pooling + L2 normalization, so
cosine distance is the right metric. Retrieval queries get an instruct prefix;
stored documents are embedded raw.
"""
from typing import Optional

import httpx
from openai import AsyncOpenAI

from . import config

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


def enabled() -> bool:
    return _enabled and _client is not None


async def embed(text: str, *, query: bool = False) -> Optional[list[float]]:
    """One embedding. Documents raw; queries get the retrieval instruct prefix."""
    if not enabled() or not text.strip():
        return None
    payload = (config.EMBED_QUERY_PROMPT + text) if query else text
    try:
        r = await _client.embeddings.create(model=config.EMBED_MODEL, input=payload)
        return list(r.data[0].embedding)
    except Exception:
        return None
