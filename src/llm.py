"""Chat LLM via an OpenAI-compatible endpoint.

No temperature and no max_tokens are ever sent — the server's own defaults
decide sampling and output length. Tool calls use the OpenAI function-calling
shape; the dialogue loop runs them until the model returns a plain message.

All calls stream: the endpoint returns SSE unconditionally, so a non-streaming
create() would hang waiting for a whole-JSON body. The SDK's .stream() helper
assembles the final message (incl. tool_calls) and usage for us.
"""
import asyncio
import json
import re
from typing import Any, Optional

import httpx
from openai import APIError, AsyncOpenAI, LengthFinishReasonError

from . import config

# The proxy multi-routes to upstreams; some transiently reject (e.g. 400 "User
# location is not supported", flaky network). A retry usually lands on a good
# route. Keep small — each call is slow.
_RETRIES = 3
_RETRY_SLEEP = 1.5


async def _complete(model: str, messages: list[dict], extra: dict) -> Any:
    """One streamed completion with retries on transient upstream/network errors."""
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            async with _client.chat.completions.stream(
                model=model, messages=messages, **extra
            ) as stream:
                try:
                    return await stream.get_final_completion()
                except LengthFinishReasonError as e:
                    # Output cap hit (server default; we send no max_tokens).
                    # Return the partial completion so callers can salvage what
                    # streamed instead of losing the whole turn.
                    return e.completion
        except (APIError, httpx.HTTPError) as e:
            last = e
            if attempt < _RETRIES - 1:
                await asyncio.sleep(_RETRY_SLEEP)
    raise last  # type: ignore[misc]

# trust_env=False: do NOT inherit the OS/registry proxy. Otherwise httpx routes
# a localhost endpoint through the system proxy and hangs (ReadTimeout).
_http = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(120.0))
_client = AsyncOpenAI(
    base_url=config.OPENAI_BASE_URL,
    api_key=config.OPENAI_API_KEY,
    http_client=_http,
)

# Max tool rounds per turn before we force a tools-off answer (loop guard).
# Each round is a full-context completion WITH reasoning, so this directly
# bounds per-turn token burn. Plan/mark/search realistically need <=2 rounds.
MAX_TOOL_ROUNDS = 4


def _usage(resp) -> int:
    u = getattr(resp, "usage", None)
    return getattr(u, "total_tokens", 0) or 0 if u is not None else 0


async def dialogue(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    apply_tool,  # async (name, args) -> str; may raise to abort the turn
) -> tuple[str, int]:
    """Run one interview turn: the model may call tools, then produces the
    question. `messages` (incl. the system message) is mutated in place with
    assistant/tool turns. Returns (final visible text, tokens spent this turn).
    """
    tokens = 0
    for round_ in range(MAX_TOOL_ROUNDS + 1):
        # On the final allowed round, drop tools so the model must answer in
        # prose — a guard against a model that loops tool calls forever.
        use_tools = round_ < MAX_TOOL_ROUNDS
        extra = {"tools": tools, "tool_choice": "auto"} if use_tools else {}
        resp = await _complete(config.DIALOG_MODEL, messages, extra)
        tokens += _usage(resp)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip(), tokens

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = await apply_tool(tc.function.name, args)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": out}
            )
    return "", tokens  # unreachable: last round runs without tools


async def chat(messages: list[dict[str, Any]], model: str = "") -> str:
    """Plain multi-turn completion, no tools. Used by the expert simulator in
    scripts/selftest.py (the model plays the interviewee)."""
    resp = await _complete(model or config.DIALOG_MODEL, messages, {})
    return (resp.choices[0].message.content or "").strip()


def _strip_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()


def _salvage_array(text: str) -> list:
    """Recover the complete top-level objects from a possibly-truncated JSON
    array (output cut off by the length cap). Returns whatever fully parsed
    before truncation; [] if nothing usable."""
    text = _strip_fence(text)
    start = text.find("[")
    if start < 0:
        return []
    dec = json.JSONDecoder()
    body = text[start + 1:]
    items: list = []
    k = 0
    n = len(body)
    while k < n:
        while k < n and body[k] in " \t\r\n,":
            k += 1
        if k >= n or body[k] == "]":
            break
        try:
            obj, end = dec.raw_decode(body, k)
        except json.JSONDecodeError:
            break  # reached the truncated (incomplete) tail
        items.append(obj)
        k = end
    return items


async def _content(model: str, system: str, user: str) -> str:
    resp = await _complete(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        {},
    )
    return resp.choices[0].message.content or ""


async def _json_call(model: str, system: str, user: str) -> Any:
    return json.loads(_strip_fence(await _content(model, system, user)))


async def extract(user: str) -> list[dict]:
    """Transcript chunk -> items. Robust to a length-capped response: if the
    JSON array is truncated, salvage the objects that fully streamed. One retry
    on total parse failure."""
    from . import prompts
    for attempt in range(2):
        text = await _content(config.EXTRACT_MODEL, prompts.EXTRACT_SYSTEM, user)
        try:
            data = json.loads(_strip_fence(text))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            salvaged = _salvage_array(text)
            if salvaged:
                return salvaged
            if attempt == 1:
                raise
    return []


async def contradiction(a: str, b: str) -> bool:
    """Cheap judge: are two nearby facts the same claim or conflicting?"""
    from . import prompts
    user = f"Утверждение A:\n{a}\n\nУтверждение B:\n{b}"
    try:
        data = await _json_call(config.EVAL_MODEL, prompts.CONTRADICTION_SYSTEM, user)
        return bool(data.get("contradict")) if isinstance(data, dict) else False
    except (json.JSONDecodeError, APIError):
        return False


async def summary(user: str) -> str:
    from . import prompts
    return (await _content(config.SUMMARY_MODEL, prompts.SUMMARY_SYSTEM, user)).strip()


async def eval_question(user: str) -> Optional[dict]:
    from . import prompts
    try:
        return await _json_call(config.EVAL_MODEL, prompts.EVAL_SYSTEM, user)
    except (json.JSONDecodeError, APIError):
        return None
