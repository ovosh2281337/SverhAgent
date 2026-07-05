"""Quick inspection of collected knowledge — the Этап-3 "look for silent data
corruption" surface. Run: python -m scripts.view [topic]
"""
import asyncio
import json
import sys

from src import db


async def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "default"
    rows = await db.all_for_topic(topic)

    print(f"=== Извлечённые элементы (тема: {topic}, всего {len(rows)}) ===\n")
    for r in rows:
        payload = r["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        tags = [r["origin"]]
        if r["duplicate_of"]:
            tags.append(f"дубль #{r['duplicate_of']}")
        else:
            tags.append(f"подтв.={r['confirmation_count']}")
        print(f"[{r['type']} | {' | '.join(tags)}] {r['expert_name']}")
        print(f"  {json.dumps(payload, ensure_ascii=False)}")
        print(f"  цитата: {r['quote']}\n")

    summary = await db.topic_summary(topic)
    print("=== Сводка по теме ===\n")
    print(summary or "(ещё не собрана)")
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
