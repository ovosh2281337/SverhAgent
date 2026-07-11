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


class DialogueContextExceeded(RuntimeError):
    """Raised before a completion whose prompt cannot fit model context."""

    def __init__(self, spent: int = 0):
        super().__init__("dialogue prompt exceeds model context before LLM request")
        self.spent = spent


def estimate_tokens(value: Any) -> int:
    """Conservative tokenizer-free estimate suitable for mixed Russian/JSON."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, (len(value) + 1) // 2)


def estimate_request_tokens(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> int:
    # Per-message framing/tool schema overhead matters for many short turns.
    return 64 + estimate_tokens(messages) + (estimate_tokens(tools) if tools else 0)


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
    rounds_out: Optional[list[dict]] = None,  # per-round trace, appended in place
) -> tuple[str, int]:
    """Run one interview turn: the model may call tools, then produces the
    question. `messages` (incl. the system message) is mutated in place with
    assistant/tool turns. Returns (final visible text, tokens spent this turn).

    If `rounds_out` is given, one dict per completion round is appended to it
    ({n, tokens, tools, final}) so verbose mode can show the model's step-by-step
    reasoning loop, not just the final tool set.
    """
    tokens = 0
    for round_ in range(MAX_TOOL_ROUNDS + 1):
        # On the final allowed round, drop tools so the model must answer in
        # prose — a guard against a model that loops tool calls forever.
        use_tools = round_ < MAX_TOOL_ROUNDS
        active_tools = tools if use_tools else []
        prompt_tokens = estimate_request_tokens(messages, active_tools)
        if prompt_tokens >= config.DIALOG_CONTEXT_TOKENS:
            raise DialogueContextExceeded(tokens)
        extra = {}
        if use_tools:
            extra.update({"tools": tools, "tool_choice": "auto"})
        resp = await _complete(config.DIALOG_MODEL, messages, extra)
        msg = resp.choices[0].message
        charged = _usage(resp) or (
            prompt_tokens + estimate_tokens(msg.content or "")
        )
        tokens += charged
        if rounds_out is not None:
            rounds_out.append({
                "n": round_ + 1,
                "tokens": charged,
                "tools": [tc.function.name for tc in (msg.tool_calls or [])],
                "final": not msg.tool_calls,
            })
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


async def ground_extraction(user: str) -> dict:
    """Adversarial semantic-grounding verdict. Malformed judge output is a
    hard failure: publishing without a valid verdict would silently bypass the
    trust boundary. One fresh completion is allowed before failing closed."""
    from . import prompts

    last_error: Exception | None = None
    for _ in range(2):
        try:
            data = await _json_call(
                config.GROUND_MODEL, prompts.GROUNDING_SYSTEM, user
            )
            if not isinstance(data, dict):
                raise ValueError("grounding verdict is not an object")
            if data.get("verdict") not in {"verified", "partial", "rejected"}:
                raise ValueError("grounding verdict is unknown")
            if not isinstance(data.get("reason"), str) or not data["reason"].strip():
                raise ValueError("grounding reason is empty")
            unsupported = data.get("unsupported_atoms")
            if not isinstance(unsupported, list) or not all(
                isinstance(atom, str) for atom in unsupported
            ):
                raise ValueError("unsupported_atoms must be a string array")
            if not isinstance(data.get("ambiguous"), bool):
                raise ValueError("grounding ambiguous must be boolean")
            return {
                "verdict": data["verdict"],
                "reason": data["reason"].strip(),
                "unsupported_atoms": [a.strip() for a in unsupported if a.strip()],
                "ambiguous": data["ambiguous"],
            }
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise ValueError("grounding judge returned invalid JSON twice") from last_error


async def repair_extraction(user: str) -> list[dict]:
    """One candidate-level repair. The caller records the original candidate
    when this returns no usable items, so malformed model output is observable
    rather than silently discarded."""
    from . import prompts

    for _ in range(2):
        text = await _content(
            config.EXTRACT_MODEL, prompts.REPAIR_EXTRACTION_SYSTEM, user
        )
        try:
            data = json.loads(_strip_fence(text))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            salvaged = _salvage_array(text)
            if salvaged:
                return salvaged
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


async def classify_memory_relation(
    statement: str, candidates: list[dict]
) -> dict:
    from . import prompts
    user = json.dumps(
        {"statement": statement, "candidates": candidates},
        ensure_ascii=False,
    )
    try:
        data = await _json_call(
            config.EVAL_MODEL, prompts.MEMORY_RELATION_SYSTEM, user
        )
    except (json.JSONDecodeError, APIError):
        return {"relation": "new", "target_item_id": None,
                "confidence": 0.0, "reason": "classifier failure"}
    allowed = {
        "duplicate_of", "supports", "contradicts", "refines", "depends_on", "new"
    }
    if not isinstance(data, dict) or data.get("relation") not in allowed:
        return {"relation": "new", "target_item_id": None,
                "confidence": 0.0, "reason": "invalid classifier output"}
    candidate_ids = {candidate["item_id"] for candidate in candidates}
    target = data.get("target_item_id")
    if data["relation"] != "new" and target not in candidate_ids:
        return {"relation": "new", "target_item_id": None,
                "confidence": 0.0, "reason": "target outside candidates"}
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "relation": data["relation"],
        "target_item_id": target if data["relation"] != "new" else None,
        "confidence": confidence,
        "reason": str(data.get("reason") or "").strip(),
    }


async def verify_memory_relation(
    relation: str, source_statement: str, target_statement: str
) -> dict:
    from . import prompts
    user = json.dumps(
        {"relation": relation, "source": source_statement, "target": target_statement},
        ensure_ascii=False,
    )
    try:
        data = await _json_call(
            config.GROUND_MODEL, prompts.MEMORY_RELATION_VERIFY_SYSTEM, user
        )
    except (json.JSONDecodeError, APIError):
        return {"verified": False, "reason": "verifier failure"}
    if not isinstance(data, dict) or not isinstance(data.get("verified"), bool):
        return {"verified": False, "reason": "invalid verifier output"}
    return {"verified": data["verified"],
            "reason": str(data.get("reason") or "").strip()}


async def extract_memory_entities(statements: list[str]) -> list[list[str]]:
    from . import prompts
    data = await _json_call(
        config.EXTRACT_MODEL,
        prompts.MEMORY_ENTITIES_SYSTEM,
        json.dumps(statements, ensure_ascii=False),
    )
    if not isinstance(data, list) or len(data) != len(statements):
        raise ValueError("entity extractor returned wrong batch shape")
    result: list[list[str]] = []
    for values in data:
        if not isinstance(values, list):
            raise ValueError("entity extractor item is not an array")
        result.append([
            value.strip() for value in values if isinstance(value, str) and value.strip()
        ][:12])
    return result


async def hierarchical_memory_summary(items: list[dict]) -> str:
    from . import prompts
    return (
        await _content(
            config.SUMMARY_MODEL,
            prompts.HIERARCHICAL_SUMMARY_SYSTEM,
            json.dumps(items, ensure_ascii=False),
        )
    ).strip()


async def summary(user: str) -> str:
    from . import prompts
    return (await _content(config.SUMMARY_MODEL, prompts.SUMMARY_SYSTEM, user)).strip()


async def compact_history(previous_summary: str, turns: list[dict]) -> str:
    from . import prompts
    user = json.dumps(
        {"previous_summary": previous_summary, "new_old_turns": turns},
        ensure_ascii=False,
    )
    return (
        await _content(
            config.SUMMARY_MODEL,
            prompts.CONTEXT_COMPACTION_SYSTEM,
            user,
        )
    ).strip()


async def eval_question(user: str) -> Optional[dict]:
    from . import prompts
    try:
        return await _json_call(config.EVAL_MODEL, prompts.EVAL_SYSTEM, user)
    except (json.JSONDecodeError, APIError):
        return None
