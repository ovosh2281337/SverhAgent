"""Question-eval job: cheap batch judge over every agent question.

Not a production gate — statistics for prompt iteration against a fixed test
set. Offline eval catches drift that in-thinking self-critique misses.
"""
import asyncio

from .. import db, llm


def _verdict_line(mid: int, v: dict) -> str:
    def flag(key: str) -> str:
        val = v.get(key)
        return "да" if val is True else "нет" if val is False else str(val)
    note = v.get("note")
    tail = f" — {note}" if note else ""
    return (
        f" • Q#{mid}: anchored={flag('anchored')} specific={flag('specific')} "
        f"banlist={flag('banlist')}{tail}"
    )


async def run(session_id: int, trace: list[str] | None = None) -> int:
    rows = await db.transcript(session_id)
    done = await db.evaluated_message_ids(session_id)

    if trace is not None:
        trace.append("🎯 Оценка вопросов (ЛЛМ-судья, по каждому вопросу агента):")
    scored = 0
    prev_expert = None  # (id, text) of the last expert answer
    for r in rows:
        if r["role"] == "user":
            prev_expert = (r["id"], r["content"])
            continue
        # assistant question
        if r["id"] in done or prev_expert is None:
            continue
        answer_id, answer_text = prev_expert
        user = (
            f"Ответ эксперта:\n{answer_text}\n\n"
            f"Следующий вопрос агента:\n{r['content']}"
        )
        verdict = await llm.eval_question(user)
        if verdict is None:
            continue
        await db.add_question_eval(r["id"], answer_id, verdict)
        if trace is not None:
            trace.append(_verdict_line(r["id"], verdict))
        scored += 1
    if trace is not None and scored == 0:
        trace.append(" • (новых вопросов для оценки нет)")
    return scored


if __name__ == "__main__":
    import sys

    async def _main():
        sid = int(sys.argv[1])
        n = await run(sid)
        print(f"scored {n} questions in session {sid}")
        await db.close()

    asyncio.run(_main())
