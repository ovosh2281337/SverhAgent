"""Telegram front end. Zero onboarding: /start begins an interview."""
import asyncio
import json
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import BotCommand, Message

from . import agent, config, db
from .jobs import eval as evaljob
from .jobs import extract, summary

_PLED = 3500  # Telegram message cap is 4096; leave headroom


def _payload_line(type_: str, payload) -> str:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if type_ == "fact":
        s = payload.get("statement", "")
        q = payload.get("qualifiers", "")
        return f"{s}" + (f" ({q})" if q else "")
    if type_ == "qa_pair":
        return f"Q: {payload.get('question','')} — A: {payload.get('answer','')}"
    if type_ == "term":
        return f"{payload.get('term','')}: {payload.get('definition','')}"
    return json.dumps(payload, ensure_ascii=False)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

dp = Dispatcher()

INTRO = (
    "Привет! Я собираю экспертное знание через короткое интервью (~15–20 минут).\n"
    "Буду задавать конкретные вопросы про вашу практику: кейсы, числа, "
    "trade-off'ы, провалы. Отвечайте свободно, как удобно.\n"
    "Если про что-то не хотите говорить — так и скажите, перейдём дальше.\n"
    "В конце покажу, что записал, — сверимся.\n\n"
    "Напишите первое сообщение (пару слов о вашей специализации), и начнём."
)


def _expert_name(msg: Message) -> str:
    u = msg.from_user
    return (u.full_name or u.username or str(u.id)) if u else "unknown"


@dp.message(Command("start"))
async def on_start(msg: Message, command: CommandObject) -> None:
    topic = (command.args or "default").strip()
    existing = await db.active_session(_expert_name(msg))
    if existing:
        await msg.answer(
            "У вас уже есть активная сессия. Продолжайте отвечать, "
            "или /finish чтобы завершить и запустить обработку."
        )
        return
    await db.create_session(_expert_name(msg), topic)
    await msg.answer(INTRO)


@dp.message(Command("finish"))
async def on_finish(msg: Message) -> None:
    sess = await db.active_session(_expert_name(msg))
    if not sess:
        await msg.answer("Активной сессии нет. /start чтобы начать.")
        return
    await db.finish_session(sess["id"])
    _locks.pop(sess["id"], None)  # finished session gets no more turns
    await msg.answer("Сессия завершена. Запускаю обработку…")
    asyncio.create_task(_postprocess(msg.bot, msg.chat.id, sess["id"], sess["topic"]))


@dp.message(Command("plan"))
async def on_plan(msg: Message) -> None:
    """Show the interview plan/progress — the expert sees the table of contents."""
    sess = await db.active_session(_expert_name(msg))
    if not sess:
        await msg.answer("Активной сессии нет. /start чтобы начать.")
        return
    items = await db.plan_items(sess["id"])
    if not items:
        await msg.answer("План ещё не составлен — появится после первых вопросов.")
        return
    lines = ["План интервью:"]
    for r in items:
        lines.append(("✅ " if r["status"] == "covered" else "▫️ ") + r["subtopic"])
    await msg.answer("\n".join(lines))


@dp.message(Command("reset"))
async def on_reset(msg: Message) -> None:
    sess = await db.active_session(_expert_name(msg))
    if not sess:
        await msg.answer("Активной сессии нет. /start чтобы начать новую.")
        return
    await db.delete_session(sess["id"])
    _locks.pop(sess["id"], None)
    await msg.answer("Сессия и переписка удалены. /start чтобы начать с нуля.")


_locks: dict[int, asyncio.Lock] = {}  # per-session, serializes concurrent turns


def _lock(session_id: int) -> asyncio.Lock:
    lock = _locks.get(session_id)
    if lock is None:
        lock = _locks[session_id] = asyncio.Lock()
    return lock


@dp.message(F.text)
async def on_text(msg: Message) -> None:
    name = _expert_name(msg)
    sess = await db.active_session(name)
    if not sess:
        await msg.answer("Нет активной сессии. Наберите /start чтобы начать.")
        return

    # Serialize turns of one session: two fast messages must not race on the
    # same context (append + run_turn + append) and interleave the history.
    # Bounded wait: if a previous turn wedged (it shouldn't — llm.py has httpx
    # timeouts — but belt and braces), the user gets an answer, not silence.
    lock = _lock(sess["id"])
    try:
        await asyncio.wait_for(lock.acquire(), timeout=240)
    except asyncio.TimeoutError:
        await msg.answer(
            "Всё ещё обрабатываю предыдущее сообщение — подождите немного "
            "и отправьте ещё раз."
        )
        return
    try:
        await db.add_message(sess["id"], "user", msg.text)
        await msg.bot.send_chat_action(msg.chat.id, "typing")
        try:
            text, finished = await agent.run_turn(sess["id"], sess["topic"])
        except Exception:
            log.exception("turn failed")
            await msg.answer("Упс, сбой на моей стороне. Попробуйте написать ещё раз.")
            return
    finally:
        lock.release()

    await msg.answer(text or "…")
    if finished:
        await msg.answer("Спасибо! Обрабатываю записанное…")
        asyncio.create_task(_postprocess(msg.bot, msg.chat.id, sess["id"], sess["topic"]))


def _recap(rows) -> str:
    if not rows:
        return (
            "Обработал сессию — извлекать в базу знаний оказалось нечего "
            "(содержательных утверждений эксперта не набралось)."
        )
    canon = [r for r in rows if not r["duplicate_of"]]
    dups = len(rows) - len(canon)
    lines = [f"Сохранил в базу знаний ({len(canon)} записей"
             + (f", +{dups} дубль-подтверждений" if dups else "") + "):", ""]
    for r in canon:
        tag = "гипотеза" if r["origin"] == "confirmed_hypothesis" else r["type"]
        line = f"• [{tag}] {_payload_line(r['type'], r['payload'])}"
        lines.append(line[:400])
    text = "\n".join(lines)
    if len(text) > _PLED:
        text = text[:_PLED] + "\n… (обрезано, полностью — scripts/view)"
    return text


async def _postprocess(bot: Bot, chat_id: int, session_id: int, topic: str) -> None:
    try:
        n = await extract.run(session_id)
        await summary.run(topic)
        log.info("postprocess done: session=%s items=%s topic=%s", session_id, n, topic)
        rows = await db.extracted_for_session(session_id)
        await bot.send_message(chat_id, _recap(rows))
        # Offline question-quality scoring — cheap judge, out of the interview
        # token budget. Never let its failure hide a successful extraction.
        try:
            scored = await evaljob.run(session_id)
            log.info("eval done: session=%s scored=%s", session_id, scored)
        except Exception:
            log.exception("eval failed session=%s", session_id)
    except Exception:
        log.exception("postprocess failed session=%s", session_id)
        try:
            await bot.send_message(chat_id, "Не смог обработать сессию — сбой на моей стороне.")
        except Exception:
            pass


async def main() -> None:
    bot = Bot(config.TELEGRAM_BOT_TOKEN)
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать интервью (/start тема)"),
        BotCommand(command="plan", description="Показать план и прогресс интервью"),
        BotCommand(command="finish", description="Завершить и обработать"),
        BotCommand(command="reset", description="Удалить активную сессию, начать с нуля"),
    ])
    try:
        await dp.start_polling(bot)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
