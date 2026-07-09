"""Telegram front end. Zero onboarding: /start begins an interview."""
import asyncio
import json
import logging
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import BotCommand, Message

from . import agent, config, db, embed, stt
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


def _telegram_user_id(msg: Message) -> int | None:
    u = msg.from_user
    return int(u.id) if u else None


def _telegram_username(msg: Message) -> str | None:
    u = msg.from_user
    return u.username if u else None


def _telegram_full_name(msg: Message) -> str | None:
    u = msg.from_user
    return u.full_name if u else None


async def _active_session_for_msg(msg: Message):
    user_id = _telegram_user_id(msg)
    if user_id is None:
        return None
    return await db.active_session_for_user(user_id)


@dp.message(Command("start"))
async def on_start(msg: Message, command: CommandObject) -> None:
    topic = (command.args or "default").strip()
    user_id = _telegram_user_id(msg)
    if user_id is None:
        await msg.answer("Не могу определить Telegram user_id. Напишите боту из обычного пользовательского аккаунта.")
        return
    existing = await db.active_session_for_user(user_id)
    if existing:
        await msg.answer(
            "У вас уже есть активная сессия. Продолжайте отвечать, "
            "или /finish чтобы завершить и запустить обработку."
        )
        return
    try:
        await db.create_session(
            _expert_name(msg),
            topic,
            telegram_user_id=user_id,
            telegram_username=_telegram_username(msg),
            telegram_full_name=_telegram_full_name(msg),
        )
    except db.ActiveSessionExistsError:
        await msg.answer(
            "У вас уже есть активная сессия. Продолжайте отвечать, "
            "или /finish чтобы завершить и запустить обработку."
        )
        return
    await msg.answer(INTRO)


@dp.message(Command("finish"))
async def on_finish(msg: Message) -> None:
    sess = await _active_session_for_msg(msg)
    if not sess:
        await msg.answer("Активной сессии нет. /start чтобы начать.")
        return
    lock = await _acquire_lifecycle_lock(
        msg,
        sess["id"],
        "Всё ещё обрабатываю предыдущее сообщение — подождите немного и отправьте /finish ещё раз.",
    )
    if lock is None:
        return
    postprocess_args = None
    no_active_after_wait = False
    try:
        fresh = await _fresh_active_session_for_msg(msg, sess["id"])
        if not fresh:
            no_active_after_wait = True
        else:
            await db.finish_session(fresh["id"])
            postprocess_args = (fresh["id"], fresh["topic"])
    finally:
        lock.release()
    if no_active_after_wait:
        await msg.answer("Активной сессии нет. /start чтобы начать.")
        return
    if postprocess_args is None:
        log.error("finish lifecycle ended without postprocess args: session=%s", sess["id"])
        await msg.answer("Не смог завершить сессию — сбой на моей стороне.")
        return
    await msg.answer("Сессия завершена. Запускаю обработку…")
    session_id, topic = postprocess_args
    asyncio.create_task(_postprocess(msg.bot, msg.chat.id, session_id, topic))


@dp.message(Command("plan"))
async def on_plan(msg: Message) -> None:
    """Show the interview plan/progress — the expert sees the table of contents."""
    sess = await _active_session_for_msg(msg)
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


@dp.message(Command("verbose"))
async def on_verbose(msg: Message) -> None:
    """Toggle debug mode: show the full pipeline (tool calls, token spend, the
    STATE context, transcriptions) after each reply. Off restores the original UX."""
    chat = msg.chat.id
    new = not _verbose.get(chat, False)
    _verbose[chat] = new
    if new:
        await msg.answer(
            "🔍 Режим отладки ВКЛЮЧЁН.\n\n"
            "После каждого ответа — что происходит «под капотом» за ход:\n"
            "• раунды модели (сколько раз думала, с какими токенами);\n"
            "• какие инструменты вызвал и с какими запросами "
            "(поиск по базе, веб-поиск, план);\n"
            "• токены за ход;\n"
            "• полный контекст (STATE) на входе — план, факты из прошлых "
            "сессий, бюджет.\n\n"
            "После /finish — вся обработка ЛЛМ по стадиям:\n"
            "• извлечение (чанки → записи) и решение дедупа по каждой "
            "(новое / дубль / противоречие, с судьёй ЛЛМ);\n"
            "• пересборка сводки по теме;\n"
            "• оценка каждого заданного вопроса.\n\n"
            "Распознанный текст голосовых показывается всегда.\n"
            "Выключить и вернуть как было: /verbose"
        )
    else:
        await msg.answer("Режим отладки выключен — вернул как было. Снова включить: /verbose")


@dp.message(Command("reset"))
async def on_reset(msg: Message) -> None:
    sess = await _active_session_for_msg(msg)
    if not sess:
        await msg.answer("Активной сессии нет. /start чтобы начать новую.")
        return
    lock = await _acquire_lifecycle_lock(
        msg,
        sess["id"],
        "Всё ещё обрабатываю предыдущее сообщение — подождите немного и отправьте /reset ещё раз.",
    )
    if lock is None:
        return
    no_active_after_wait = False
    try:
        fresh = await _fresh_active_session_for_msg(msg, sess["id"])
        if not fresh:
            no_active_after_wait = True
        else:
            await db.delete_session(fresh["id"])
    finally:
        lock.release()
    if no_active_after_wait:
        await msg.answer("Активной сессии нет. /start чтобы начать новую.")
        return
    await msg.answer("Сессия и переписка удалены. /start чтобы начать с нуля.")


_LIFECYCLE_LOCK_TIMEOUT_SEC = 240
_locks: dict[int, asyncio.Lock] = {}
"""Per-session lifecycle locks.

The same lock serializes user turns, /finish and /reset. Do not remove a lock
immediately after finish/reset: stale handlers may already hold the old session
id and must queue on the same lock, then re-read DB state and abort safely.
"""


def _lock(session_id: int) -> asyncio.Lock:
    lock = _locks.get(session_id)
    if lock is None:
        lock = _locks[session_id] = asyncio.Lock()
    return lock


async def _acquire_lifecycle_lock(
    msg: Message, session_id: int, busy_text: str
) -> asyncio.Lock | None:
    lock = _lock(session_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LIFECYCLE_LOCK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        await msg.answer(busy_text)
        return None
    return lock


async def _fresh_active_session_for_msg(msg: Message, session_id: int):
    """Re-read session after waiting for the lifecycle lock.

    Handlers first find an active session by Telegram user_id, but they may wait
    behind an in-flight turn. By the time they acquire the lock, that session
    may have been finished or deleted. This check prevents stale handlers from
    mutating a non-active or no-longer-owned session.
    """
    user_id = _telegram_user_id(msg)
    if user_id is None:
        return None
    sess = await db.get_session(session_id)
    if not sess:
        return None
    if sess["status"] != "active":
        return None
    if sess["telegram_user_id"] != user_id:
        return None
    return sess


# Verbose / debug mode: per chat, off by default (off == original behavior, so
# turning it off literally restores "как было"). In-memory: a restart resets to
# the default-off state, which is the safe baseline. Toggle with /verbose.
_verbose: dict[int, bool] = {}

_TOOL_ICON = {
    "web_search": "🔎", "web_fetch": "🌐", "search_knowledge": "📚",
    "update_plan": "🗂", "mark_covered": "✅", "end_session": "🏁",
}


def _tool_line(call: dict) -> str:
    name = call.get("name", "?")
    args = call.get("args") or {}
    icon = _TOOL_ICON.get(name, "🛠")
    if name in ("web_search", "search_knowledge"):
        detail = f"«{args.get('query', '')}»"
    elif name == "web_fetch":
        detail = args.get("url", "")
    elif name == "update_plan":
        subs = [s for s in (args.get("subtopics") or []) if s]
        detail = " · ".join(subs) if subs else "(пусто)"
    elif name == "mark_covered":
        detail = args.get("subtopic", "")
    elif name == "end_session":
        detail = "завершение сессии"
    else:
        detail = json.dumps(args, ensure_ascii=False)
    return f" • {icon} {name}: {detail}"


def _trace_text(trace: dict) -> str:
    """Human-readable dump of one turn's internals for verbose mode: which tools
    the agent called (with their queries), the token spend, and the full STATE
    context it was fed (plan, cross-session RAG facts, budget)."""
    tools = trace.get("tools") or []
    lines = ["🔍 Под капотом (этот ход)", ""]
    rounds = trace.get("rounds") or []
    if rounds:
        lines.append(f"🔁 Раундов модели: {len(rounds)}")
        for r in rounds:
            what = "ответ" if r.get("final") else ", ".join(r.get("tools") or []) or "—"
            lines.append(f" • раунд {r.get('n')}: {what} · {r.get('tokens')} ток")
        lines.append("")
    if tools:
        lines.append("🛠 Вызовы инструментов:")
        lines += [_tool_line(t) for t in tools]
    else:
        lines.append("🛠 Инструменты не вызывались — ответ напрямую.")
    spent, total = trace.get("spent"), trace.get("total")
    if spent is not None:
        lines += ["", f"📊 Токенов за ход: {spent} · всего в сессии: {total}"]
    state = (trace.get("state") or "").strip()
    if state:
        lines += ["", "📥 Контекст, поданный модели (STATE):", "", state]
    return "\n".join(lines)


def _chunk_msg(text: str) -> list[str]:
    """Split text into Telegram-cap-sized pieces on line boundaries (nothing cut
    mid-line); a single overlong line is hard-sliced."""
    out: list[str] = []
    chunk = ""
    for line in text.split("\n"):
        while len(line) > _PLED:
            if chunk:
                out.append(chunk)
                chunk = ""
            out.append(line[:_PLED])
            line = line[_PLED:]
        if len(chunk) + len(line) + 1 > _PLED:
            out.append(chunk)
            chunk = line
        else:
            chunk = f"{chunk}\n{line}" if chunk else line
    if chunk:
        out.append(chunk)
    return out


async def _send_long(msg: Message, text: str) -> None:
    for c in _chunk_msg(text):
        await msg.answer(c)


async def _send_long_chat(bot: Bot, chat_id: int, text: str) -> None:
    """Chat-id variant (postprocess runs without a Message to reply to)."""
    for c in _chunk_msg(text):
        await bot.send_message(chat_id, c)


async def _expert_turn(msg: Message, sess, text: str) -> None:
    """Record one expert message, run the agent turn, reply. Shared by the text
    and voice handlers — voice differs only in how `text` was obtained."""
    # Serialize the full session lifecycle: two fast messages must not interleave
    # history, and /finish or /reset must not change state while run_turn is
    # appending assistant output.
    lock = await _acquire_lifecycle_lock(
        msg,
        sess["id"],
        "Всё ещё обрабатываю предыдущее сообщение — подождите немного и отправьте ещё раз.",
    )
    if lock is None:
        return
    stale_text = None
    error_text = None
    reply = ""
    finished = False
    trace: dict = {}
    try:
        fresh = await _fresh_active_session_for_msg(msg, sess["id"])
        if not fresh:
            stale_text = "Сессия уже завершена или удалена. Наберите /start чтобы начать новую."
        else:
            await db.add_message(fresh["id"], "user", text)
            await msg.bot.send_chat_action(msg.chat.id, "typing")
            try:
                reply, finished, trace = await agent.run_turn(fresh["id"], fresh["topic"])
            except Exception:
                log.exception("turn failed")
                error_text = "Упс, сбой на моей стороне. Попробуйте написать ещё раз."
    finally:
        lock.release()

    if stale_text:
        await msg.answer(stale_text)
        return
    if error_text:
        await msg.answer(error_text)
        return
    await msg.answer(reply or "…")
    if _verbose.get(msg.chat.id):
        await _send_long(msg, _trace_text(trace))
    if finished:
        await msg.answer("Спасибо! Обрабатываю записанное…")
        asyncio.create_task(_postprocess(msg.bot, msg.chat.id, fresh["id"], fresh["topic"]))


@dp.message(F.text)
async def on_text(msg: Message) -> None:
    sess = await _active_session_for_msg(msg)
    if not sess:
        await msg.answer("Нет активной сессии. Наберите /start чтобы начать.")
        return
    await _expert_turn(msg, sess, msg.text)


@dp.message(F.voice)
async def on_voice(msg: Message) -> None:
    sess = await _active_session_for_msg(msg)
    if not sess:
        await msg.answer("Нет активной сессии. Наберите /start чтобы начать.")
        return
    if not stt.enabled():
        await msg.answer("Голосовые пока не подключены — напишите текстом, пожалуйста.")
        return
    if msg.voice.duration and msg.voice.duration > config.STT_MAX_VOICE_SEC:
        limit_min = config.STT_MAX_VOICE_SEC // 60
        await msg.answer(
            f"Слишком длинное голосовое (лимит ~{limit_min} мин) — "
            "разбейте на части, пожалуйста."
        )
        return

    await msg.bot.send_chat_action(msg.chat.id, "typing")
    buf = BytesIO()
    try:
        await msg.bot.download(msg.voice.file_id, destination=buf)
    except Exception:
        log.exception("voice download failed")
        await msg.answer("Не смог скачать голосовое — попробуйте ещё раз или напишите текстом.")
        return

    text = await stt.transcribe(buf.getvalue())
    if text is None:
        await msg.answer("Не смог обработать голосовое — сбой распознавания, напишите текстом.")
        return
    if not text:
        await msg.answer("Не расслышал — попробуйте ещё раз или напишите текстом.")
        return

    # Show what was transcribed so the expert can correct it in the next message.
    shown = text if len(text) <= _PLED else text[:_PLED] + " …"
    await msg.answer(f"🎙 Распознал: «{shown}»")
    await _expert_turn(msg, sess, text)


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
        mode = f"; support={r['support_mode']}" if r["support_mode"] else ""
        line = f"• [{tag}{mode}] {_payload_line(r['type'], r['payload'])}"
        lines.append(line[:400])
    text = "\n".join(lines)
    if len(text) > _PLED:
        text = text[:_PLED] + "\n… (обрезано, полностью — scripts/view)"
    return text


async def _postprocess(bot: Bot, chat_id: int, session_id: int, topic: str) -> None:
    # In verbose mode the whole post-finish LLM pipeline (extract → dedup judge →
    # summary → eval) is surfaced stage by stage; a fresh list per stage is sent
    # as its own message so the expert sees exactly what the model did, not just
    # the final recap. `None` (default) keeps the jobs silent for CLI/selftest.
    vb = _verbose.get(chat_id)
    try:
        tr: list[str] | None = [] if vb else None
        n = await extract.run(session_id, tr)
        if tr:
            await _send_long_chat(bot, chat_id, "\n".join(tr))
        tr = [] if vb else None
        await summary.run(topic, tr)
        if tr:
            await _send_long_chat(bot, chat_id, "\n".join(tr))
        log.info("postprocess done: session=%s items=%s topic=%s", session_id, n, topic)
        rows = await db.extracted_for_session(session_id)
        await bot.send_message(chat_id, _recap(rows))
        # Offline question-quality scoring — cheap judge, out of the interview
        # token budget. Never let its failure hide a successful extraction.
        try:
            tr = [] if vb else None
            scored = await evaljob.run(session_id, tr)
            if tr:
                await _send_long_chat(bot, chat_id, "\n".join(tr))
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
    try:
        applied = await db.migrate()  # bring schema up to date before serving
        if applied:
            log.info("migrations applied: %s", ", ".join(applied))
        await db.vector_health_check()
        if embed.enabled():
            health = await embed.health_check()
            log.info("embedding startup health check passed: %s", health)
        else:
            log.warning("embeddings intentionally disabled: EMBED_MODE=%s", config.EMBED_MODE)
        bot = Bot(config.TELEGRAM_BOT_TOKEN)
        await bot.set_my_commands([
            BotCommand(command="start", description="Начать интервью (/start тема)"),
            BotCommand(command="plan", description="Показать план и прогресс интервью"),
            BotCommand(command="finish", description="Завершить и обработать"),
            BotCommand(command="verbose", description="Показать/скрыть внутреннюю работу (отладка)"),
            BotCommand(command="reset", description="Удалить активную сессию, начать с нуля"),
        ])
        await dp.start_polling(bot)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
