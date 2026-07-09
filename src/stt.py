"""Speech-to-text client — posts voice audio to the local GigaAM service.

Optional: without STT_BASE_URL enabled() is False and the bot asks experts to
type instead. Every error (service down, timeout, bad audio) is swallowed into
None so a flaky transcription never crashes a turn — the bot handles None as
"couldn't transcribe" and keeps the interview going.

trust_env=False: the service is on localhost; the system proxy must not be
consulted for it (same rule as llm.py / embed.py — see project notes).
"""
import logging

import httpx

from . import config

log = logging.getLogger("stt")

_enabled = bool(config.STT_BASE_URL)
# Long read timeout: a minutes-long voice message chunked through the model can
# take a while; connect stays short so a dead service fails fast.
_client = (
    httpx.AsyncClient(
        base_url=config.STT_BASE_URL,
        trust_env=False,
        timeout=httpx.Timeout(300.0, connect=5.0),
    )
    if _enabled
    else None
)


def enabled() -> bool:
    return _enabled and _client is not None


async def transcribe(data: bytes) -> str | None:
    """Transcribe voice audio bytes to text. None on any failure.

    Returns "" (empty string) when the service ran but found no speech — the
    caller distinguishes that ("couldn't hear you") from None ("service down")."""
    if not enabled() or not data:
        return None
    try:
        r = await _client.post("/v1/transcribe", content=data)
        r.raise_for_status()
        return (r.json().get("text") or "").strip()
    except Exception:
        log.exception("stt transcribe failed")
        return None
