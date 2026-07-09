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


async def run(topic: str, trace: list[str] | None = None) -> bool:
    rows = await db.canonical_for_topic(topic)
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
    await db.upsert_topic_summary(topic, text)
    if trace is not None:
        trace.append(
            f"📝 Сводка по теме «{topic}»: пересобрана из {len(rows)} "
            f"канонических фактов ({len(text)} симв)."
        )
    return True


if __name__ == "__main__":
    import sys

    async def _main():
        topic = sys.argv[1] if len(sys.argv) > 1 else "default"
        ok = await run(topic)
        print(f"summary rebuilt: {ok} (topic={topic})")
        await db.close()

    asyncio.run(_main())
