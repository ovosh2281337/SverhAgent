"""Quick inspection of collected knowledge — the Этап-3 "look for silent data
corruption" surface. Run: python -m scripts.view [topic]
"""
import asyncio
import json
import sys

from src import db


def _json(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        return json.loads(value)
    return value


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m scripts.view WORKSPACE_ID TOPIC_ID")
    workspace_id, topic_id = int(sys.argv[1]), int(sys.argv[2])
    topic_row = await db.get_topic(workspace_id, topic_id)
    if topic_row is None:
        raise SystemExit("topic does not belong to workspace")
    topic = topic_row["name"]
    rows = await db.all_for_topic(workspace_id, topic_id)

    print(f"=== Извлечённые элементы (тема: {topic}, всего {len(rows)}) ===\n")
    for r in rows:
        payload = _json(r["payload"], {})
        tags = [r["origin"], f"grounding={r['grounding_status']}"]
        if r["support_mode"]:
            tags.append(f"support={r['support_mode']}")
        if r["duplicate_of"]:
            tags.append(f"дубль #{r['duplicate_of']}")
        else:
            tags.append(f"подтв.={r['confirmation_count']}")
        print(f"[{r['type']} | {' | '.join(tags)}] {r['expert_name']}")
        print(f"  {json.dumps(payload, ensure_ascii=False)}")
        details = _json(r["grounding_details"], {})
        if details:
            print(f"  grounding_details: {json.dumps(details, ensure_ascii=False)}")
        provenance = _json(r["provenance"], [])
        if provenance:
            print("  provenance:")
            for span in provenance:
                print(
                    f"    - #{span.get('message_id')} {span.get('kind')}: "
                    f"{span.get('quote')}"
                )
        else:
            print(f"  legacy_quote: {r['quote']}")
        print()

    summary = await db.topic_summary(workspace_id, topic_id)
    print("=== Сводка по теме ===\n")
    print(summary or "(ещё не собрана)")
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
