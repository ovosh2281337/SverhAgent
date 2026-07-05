"""Topic-summary job: rebuild the cross-session summary from scratch.

Never incremental — incremental compression degrades like a re-saved JPEG and
loses the qualifiers, the most valuable part of expert knowledge. Reads only
canonical items (duplicate_of IS NULL), with origin + confirmation_count.
"""
import asyncio
import json

from .. import db, llm


def _render_items(rows) -> str:
    out = []
    for r in rows:
        payload = r["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        origin = r["origin"]
        out.append(
            f"- ({r['type']}, origin={origin}, "
            f"confirmation_count={r['confirmation_count']}, "
            f"эксперт={r['expert_name']})\n"
            f"  payload={json.dumps(payload, ensure_ascii=False)}\n"
            f"  quote: {r['quote']}"
        )
    return "\n".join(out)


async def run(topic: str) -> bool:
    rows = await db.canonical_for_topic(topic)
    if not rows:
        return False
    user = (
        f"Тема: {topic}\n\n"
        f"Канонические элементы (собери сводку С НУЛЯ из них):\n\n"
        + _render_items(rows)
    )
    text = await llm.summary(user)
    await db.upsert_topic_summary(topic, text)
    return True


if __name__ == "__main__":
    import sys

    async def _main():
        topic = sys.argv[1] if len(sys.argv) > 1 else "default"
        ok = await run(topic)
        print(f"summary rebuilt: {ok} (topic={topic})")
        await db.close()

    asyncio.run(_main())
