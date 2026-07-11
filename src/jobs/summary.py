"""Topic-summary job: rebuild the cross-session summary from scratch.

Never incremental — incremental compression degrades like a re-saved JPEG and
loses the qualifiers, the most valuable part of expert knowledge. Reads only
canonical items (duplicate_of IS NULL), with origin + confirmation_count.
"""
import asyncio
import json

from .. import db, llm


def _json(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        return json.loads(value)
    return value


def _field(row, name: str, default=None):
    try:
        return row[name]
    except (KeyError, TypeError):
        return default


def _render_supports(row) -> str:
    supports = _json(_field(row, "supports"), [])
    if not supports:
        quote = (row["quote"] or "").strip()
        return f"  support: {quote}" if quote else "  support: (нет)"
    lines = []
    for s in supports[:4]:
        quote = (s.get("quote") or "").strip()
        if not quote:
            continue
        who = s.get("expert_name") or row["expert_name"]
        kind = s.get("kind") or "support"
        lines.append(f"  support[{kind}; {who}]: {quote}")
    return "\n".join(lines) if lines else "  support: (нет)"


def _render_items(rows) -> str:
    out = []
    for r in rows:
        payload = _json(r["payload"], {})
        origin = r["origin"]
        out.append(
            f"- ({r['type']}, origin={origin}, "
            f"support_mode={r['support_mode']}, "
            f"confirmation_count={r['confirmation_count']}, "
            f"эксперт={r['expert_name']})\n"
            f"  payload={json.dumps(payload, ensure_ascii=False)}\n"
            f"{_render_supports(r)}"
        )
    return "\n".join(out)


async def run(
    workspace_id: int, topic_id: int, trace: list[str] | None = None
) -> bool:
    topic_row = await db.get_topic(workspace_id, topic_id)
    if topic_row is None:
        raise ValueError("topic does not belong to workspace")
    topic = topic_row["name"]
    rows = await db.canonical_for_topic(workspace_id, topic_id)
    if not rows:
        if trace is not None:
            trace.append("📝 Сводка: канонических фактов нет — пропущено.")
        return False
    user = (
        f"Тема: {topic}\n\n"
        f"Канонические элементы (собери сводку С НУЛЯ из них):\n\n"
        + _render_items(rows)
    )
    text = await llm.summary(user)
    await db.upsert_topic_summary(workspace_id, topic_id, text)
    if trace is not None:
        trace.append(
            f"📝 Сводка по теме «{topic}»: пересобрана из {len(rows)} "
            f"канонических фактов ({len(text)} симв)."
        )
    return True


if __name__ == "__main__":
    import sys

    async def _main():
        if len(sys.argv) < 3:
            raise SystemExit("usage: python -m src.jobs.summary WORKSPACE_ID TOPIC_ID")
        workspace_id, topic_id = int(sys.argv[1]), int(sys.argv[2])
        ok = await run(workspace_id, topic_id)
        print(f"summary rebuilt: {ok} (workspace={workspace_id}, topic={topic_id})")
        await db.close()

    asyncio.run(_main())
