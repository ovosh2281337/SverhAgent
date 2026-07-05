"""Per-session observability: cost, tool usage, and question-quality scores.

Answers "which tools did the agent use, did it help, how expensive was the
chat" from persisted data. Run: python -m scripts.stats
"""
import asyncio
import json
from collections import Counter

from src import db


async def main() -> None:
    p = await db.pool()
    sessions = await p.fetch(
        "SELECT id, topic, status, tokens_used, prompt_version FROM sessions ORDER BY id"
    )
    print("=== Сессии ===\n")
    for s in sessions:
        sid = s["id"]
        users = await p.fetchval(
            "SELECT count(*) FROM messages WHERE session_id=$1 AND role='user'", sid
        )
        # tool calls (persisted on assistant messages)
        tc_rows = await p.fetch(
            "SELECT tool_calls FROM messages "
            "WHERE session_id=$1 AND tool_calls IS NOT NULL",
            sid,
        )
        tools = Counter()
        for r in tc_rows:
            calls = r["tool_calls"]
            if isinstance(calls, str):
                calls = json.loads(calls)
            for c in calls:
                tools[c.get("name", "?")] += 1
        # question-quality verdicts
        verdicts = await p.fetch(
            "SELECT verdict FROM question_evals qe "
            "JOIN messages m ON m.id = qe.message_id WHERE m.session_id=$1",
            sid,
        )
        spec = anch = ban = tot = 0
        for v in verdicts:
            d = v["verdict"]
            if isinstance(d, str):
                d = json.loads(d)
            tot += 1
            spec += 1 if d.get("specific") else 0
            anch += 1 if d.get("anchored") else 0
            ban += 1 if d.get("banlist") else 0
        items = await p.fetchval(
            "SELECT count(*) FROM extracted_items WHERE session_id=$1", sid
        )
        per = s["tokens_used"] // max(users, 1)
        print(
            f"#{sid} [{s['topic']}] {s['status']} ({s['prompt_version']})\n"
            f"    обменов={users}  токенов={s['tokens_used']} (~{per}/обмен)  "
            f"извлечено={items}"
        )
        print(f"    туллы: {dict(tools) or '—'}")
        if tot:
            print(
                f"    качество вопросов: специфичных {spec}/{tot}, "
                f"заякорено {anch}/{tot}, в бан-листе {ban}/{tot}"
            )
        print()
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
