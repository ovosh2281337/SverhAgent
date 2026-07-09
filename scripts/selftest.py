"""Self-test: drive a full interview with an LLM-simulated expert.

The same chat model plays the interviewee (persona below), the real agent code
(agent.run_turn — prompts, tools, web search, auto-RAG, budgets) runs the
interviewer side. Lets us smoke-test a prompt change end to end without a human
expert, and regression-check against the known failure modes: rabbit holes,
invented premises, challenge conveyors, boring/googlable questions.

Sessions go to their own topic (default "selftest") so postprocessing never
pollutes the real knowledge base. Run:

    python -m scripts.selftest                 # 8 turns, persona fdm
    python -m scripts.selftest --turns 12 --postprocess
    python -m scripts.selftest --persona brew  # a second domain
"""
import argparse
import asyncio
import json
import time

from src import agent, db, llm
from src.jobs import eval as evaljob
from src.jobs import extract, summary

# The simulated expert deliberately includes the traps that broke sessions 5-8:
# terse answers on implementation minutiae (rabbit-hole bait), one dubious claim
# (skepticism-budget bait), niche hardware names (web_search bait), and no
# named company/production (invented-setup bait).
PERSONAS = {
    "fdm": (
        "Ты играешь эксперта, которого интервьюирует бот. Твой персонаж:\n"
        "инженер, несколько лет печатает функциональные детали на FDM-принтерах "
        "(Bambu Lab X1C и старый Ender 3), мелкие партии на заказ. Работает "
        "один, мастерская в гараже, компании/производства НЕТ.\n\n"
        "Опыт персонажа (отвечай из него, не выдумывай энциклопедию):\n"
        "- Основной материал PETG, для нагруженных деталей PA-CF (сушка "
        "обязательна: 12 часов при 70C, иначе сопли и хрупкость).\n"
        "- Больная тема: межслойная адгезия на PA-CF, решал температурой камеры "
        "и обдувом 20%; брак упал примерно с 30% до 5%.\n"
        "- Сопла: сталь для карбона, латунь убивается за одну катушку.\n"
        "- Один спорный тезис, назови его когда-нибудь как факт: «PETG вообще "
        "не впитывает влагу, сушить его бессмысленно» (на самом деле впитывает; "
        "если бот вежливо усомнится один раз — признай, что перегнул, сушишь "
        "перед ответственными деталями).\n\n"
        "Как отвечать:\n"
        "- Как живой человек в телеграме: 1-4 предложения, без списков.\n"
        "- На вопросы про реализационную мелочь (какой разъём, куда файл "
        "положить, пункт меню) отвечай односложно и без энтузиазма.\n"
        "- На вопросы про твои решения, провалы, числа — отвечай охотно, "
        "с деталями.\n"
        "- Если вопрос общий/гуглимый («какие основные вызовы...») — ответь "
        "коротко и скучно.\n"
        "- Если бот приписывает тебе компанию/производство/команду — поправь: "
        "ты один в гараже.\n"
        "- Не задавай боту вопросов, ты отвечающий.\n"
        "Первым сообщением коротко представься (одна-две фразы о специализации)."
    ),
    "brew": (
        "Ты играешь эксперта, которого интервьюирует бот. Персонаж: обжарщик "
        "кофе, работает на ростере Aillio Bullet R1 V2, партии по 800 г, "
        "продаёт через телеграм-канал. Компании нет, арендует угол в цеху.\n"
        "- Больная тема: первый крэк на светлой обжарке, недожар — травянистость.\n"
        "- Спорный тезис (назови как факт): «датчик RoR на Bullet бесполезен, "
        "его все отключают» (если бот усомнится один раз — уточни, что не "
        "бесполезен, а шумный на малых партиях).\n"
        "- На вопросы про кнопки/меню ростера отвечай односложно.\n"
        "Отвечай как живой человек в телеграме: 1-4 предложения. Не задавай "
        "боту вопросов. Первым сообщением коротко представься."
    ),
}

_SIM_REMINDER = (
    "(Напоминание роли: отвечай как твой персонаж, 1-4 предложения, "
    "односложно на вопросы про мелочи.)"
)


async def _simulate_reply(persona: str, session_id: int) -> str:
    """Next expert message: replay the dialogue with roles flipped, so the
    simulator answers the agent's latest question in character."""
    rows = await db.history(session_id)
    messages = [{"role": "system", "content": persona}]
    for r in rows:
        role = "assistant" if r["role"] == "user" else "user"
        messages.append({"role": role, "content": r["content"]})
    if messages[-1]["role"] == "user":
        messages[-1]["content"] += f"\n\n{_SIM_REMINDER}"
    return await llm.chat(messages)


async def _last_tool_calls(session_id: int) -> list[str]:
    p = await db.pool()
    row = await p.fetchrow(
        "SELECT tool_calls FROM messages WHERE session_id=$1 AND "
        "role='assistant' ORDER BY id DESC LIMIT 1",
        session_id,
    )
    if not row or not row["tool_calls"]:
        return []
    calls = row["tool_calls"]
    if isinstance(calls, str):
        calls = json.loads(calls)
    return [c.get("name", "?") for c in calls]


async def run(topic: str, persona_key: str, turns: int, postprocess: bool) -> None:
    persona = PERSONAS[persona_key]
    name = f"selftest-{persona_key}-{int(time.time())}"
    sess = await db.create_session(name, topic)
    sid = sess["id"]
    print(f"=== selftest: session={sid} topic={topic} persona={persona_key} ===\n")

    # The expert opens (mirrors the real flow: INTRO asks for a first message).
    expert_text = await llm.chat([
        {"role": "system", "content": persona},
        {"role": "user", "content": (
            "Привет! Я собираю экспертное знание через интервью. Напишите пару "
            "слов о вашей специализации, и начнём."
        )},
    ])

    finished = False
    for i in range(1, turns + 1):
        print(f"--- ход {i} ---")
        print(f"ЭКСПЕРТ: {expert_text}\n")
        await db.add_message(sid, "user", expert_text)
        t0 = time.monotonic()
        text, finished, _ = await agent.run_turn(sid, topic)
        dt = time.monotonic() - t0
        used = await _last_tool_calls(sid)
        tools_note = f"  [тулы: {', '.join(used)}]" if used else ""
        print(f"АГЕНТ ({dt:.0f}с){tools_note}: {text}\n")
        if finished:
            print("(агент завершил сессию)\n")
            break
        expert_text = await _simulate_reply(persona, sid)

    if not finished:
        await db.finish_session(sid)
        print("(лимит ходов — сессия закрыта принудительно)\n")

    total = await db.session_tokens(sid)
    n_msgs = len(await db.history(sid))
    print(f"итог: токенов={total}, сообщений={n_msgs}")

    if postprocess:
        n = await extract.run(sid)
        await summary.run(topic)
        scored = await evaljob.run(sid)
        p = await db.pool()
        audit = await p.fetchrow(
            "SELECT count(*) FILTER (WHERE grounding_status='verified') AS verified, "
            "       count(*) FILTER (WHERE grounding_status IN ('partial','needs_review')) AS review "
            "FROM extracted_items WHERE session_id=$1",
            sid,
        )
        rejections = await p.fetchval(
            "SELECT count(*) FROM extraction_rejections WHERE session_id=$1", sid
        )
        print(
            f"извлечено={n}, "
            f"verified={audit['verified'] if audit else 0}, "
            f"review={audit['review'] if audit else 0}, "
            f"rejections={rejections}, "
            f"вопросов оценено={scored}"
        )
        rows = await db.extracted_for_session(sid)
        for r in rows:
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            dup = f" (дубль #{r['duplicate_of']})" if r["duplicate_of"] else ""
            print(f"  [{r['type']}]{dup} {json.dumps(payload, ensure_ascii=False)[:200]}")

    print(f"\nтранскрипт/оценки: python -m scripts.stats  |  сессия #{sid}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default="selftest",
                    help="тема сессии (по умолчанию 'selftest' — не трогает боевую базу)")
    ap.add_argument("--persona", default="fdm", choices=sorted(PERSONAS))
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--postprocess", action="store_true",
                    help="после сессии прогнать extract/summary/eval")

    args = ap.parse_args()

    async def _main():
        try:
            await run(args.topic, args.persona, args.turns, args.postprocess)
        finally:
            await db.close()

    asyncio.run(_main())


if __name__ == "__main__":
    main()
