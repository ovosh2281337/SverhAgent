"""Persistent dialogue compaction and bounded prompt projection."""
import json
import logging

from . import config, db, llm

log = logging.getLogger("dialogue-context")

# Internal mechanics, not product/session limits. Trigger is token-derived; no
# fixed number of recent messages and no maximum dialogue length.
_COMPACTION_BATCH = 8
_MAX_PASSES_PER_TURN = 4
_RAW_TAIL_RATIO = 0.55


def _row_dict(row) -> dict:
    return {"id": row["id"], "role": row["role"], "content": row["content"]}


def _bounded_batches(rows: list[object]) -> list[list[dict]]:
    """Split compactor input without dropping any immutable message text.

    Oversized single messages are sliced across calls.  The database cursor is
    advanced only after every slice succeeds, so retries may repeat work but
    can never lose the unprocessed suffix from the prompt projection.
    """
    budget_chars = max(1_000, int(config.DIALOG_CONTEXT_TOKENS * 2 * 0.55))
    batches: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for row in rows:
        item = _row_dict(row)
        content = item["content"]
        offset = 0
        while offset < len(content) or (not content and offset == 0):
            remaining = budget_chars - used
            if remaining <= 0:
                batches.append(current)
                current, used = [], 0
                remaining = budget_chars
            take = min(len(content) - offset, remaining)
            part = content[offset:offset + take]
            if len(content) > budget_chars:
                part = (
                    f"[message {item['id']} chars {offset}:{offset + take}]\n"
                    + part
                )
            current.append({**item, "content": part})
            used += take
            offset += take
            if not content:
                offset = 1
            if offset < len(content):
                batches.append(current)
                current, used = [], 0
    if current:
        batches.append(current)
    return batches


async def build(session_id: int) -> tuple[list[object], str, bool]:
    """Return bounded raw tail, persistent summary and backlog marker."""
    saved = await db.context_compaction(session_id)
    summary = saved["summary"] if saved else ""
    through = int(saved["through_message_id"]) if saved else 0
    target_chars = int(config.DIALOG_CONTEXT_TOKENS * 2 * _RAW_TAIL_RATIO)

    for _ in range(_MAX_PASSES_PER_TURN):
        size = await db.uncompacted_history_size(session_id, through)
        if int(size["chars"]) + len(summary) <= target_chars:
            break
        chunk = await db.history_after(
            session_id, through, _COMPACTION_BATCH
        )
        if not chunk:
            break
        batches = _bounded_batches(chunk)
        if not batches:
            break
        try:
            pending_summary = summary
            for bounded in batches:
                pending_summary = await llm.compact_history(
                    pending_summary, bounded
                )
                if not pending_summary.strip():
                    raise ValueError("dialogue compaction returned empty")
        except Exception:
            log.exception("dialogue compaction failed: session=%s", session_id)
            break
        summary = pending_summary
        through = int(chunk[-1]["id"])
        await db.upsert_context_compaction(session_id, summary, through)

    remaining = await db.uncompacted_history_size(session_id, through)
    tail_budget = max(1_000, target_chars - len(summary))
    backlog = int(remaining["chars"]) > tail_budget
    if backlog:
        rows = await db.recent_history_within_chars(
            session_id, through, tail_budget
        )
    else:
        rows = await db.history_after_all(session_id, through)
    return rows, summary, backlog


def memory_block(summary: str, backlog: bool) -> str:
    if not summary and not backlog:
        return ""
    parts = ["<SESSION_MEMORY>"]
    if summary:
        parts.append(summary.strip())
    if backlog:
        parts.append(
            "[Compaction догоняет старую историю; последние ходы и STATE актуальны.]"
        )
    parts.append("</SESSION_MEMORY>")
    return "\n".join(parts)


def debug_json(rows: list[object]) -> str:
    return json.dumps([_row_dict(row) for row in rows], ensure_ascii=False)
