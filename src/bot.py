"""Telegram front end. Zero onboarding: /start begins an interview."""
import asyncio
import json
import logging
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import BotCommand, Message

from . import agent, config, db, embed, stt
from .jobs import worker

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
    return await db.open_session_for_user(user_id)


def _is_private(msg: Message) -> bool:
    return getattr(msg.chat, "type", "private") == "private"


@dp.message(F.chat.type != "private")
async def reject_group_chat(msg: Message) -> None:
    await msg.answer("Групповые чаты отключены: модель доступа пока только для private chat.")


@dp.message(Command("start"))
async def on_start(msg: Message, command: CommandObject) -> None:
    if not _is_private(msg):
        return
    topic = (command.args or "default").strip()
    user_id = _telegram_user_id(msg)
    if user_id is None:
        await msg.answer("Не могу определить Telegram user_id. Напишите боту из обычного пользовательского аккаунта.")
        return
    existing = await db.open_session_for_user(user_id)
    if existing:
        await msg.answer(
            "У вас уже есть активная сессия. Продолжайте отвечать, "
            "или /finish чтобы завершить и запустить обработку."
        )
        return
    try:
        access = await db.ensure_user_workspace(
            user_id, _telegram_username(msg), _telegram_full_name(msg)
        )
        topic_row = await db.resolve_topic(access["workspace_id"], topic)
        await db.create_session(
            _expert_name(msg),
            topic,
            telegram_user_id=user_id,
            telegram_username=_telegram_username(msg),
            telegram_full_name=_telegram_full_name(msg),
            user_id=access["user_id"],
            workspace_id=access["workspace_id"],
            topic_id=topic_row["id"],
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
    if sess["status"] == "draft_review":
        await _show_review(msg, sess["id"])
        return
    lock = await _acquire_lifecycle_lock(
        msg,
        sess["id"],
        "Всё ещё обрабатываю предыдущее сообщение — подождите немного и отправьте /finish ещё раз.",
    )
    if lock is None:
        return
    review_ready = False
    no_active_after_wait = False
    try:
        fresh = await _fresh_active_session_for_msg(msg, sess["id"])
        if not fresh:
            no_active_after_wait = True
        else:
            review_ready = await db.begin_review(fresh["id"])
    finally:
        lock.release()
    if no_active_after_wait:
        await msg.answer("Активной сессии нет. /start чтобы начать.")
        return
    if not review_ready:
        log.error("finish lifecycle did not enter draft_review: session=%s", sess["id"])
        await msg.answer("Не смог завершить сессию — сбой на моей стороне.")
        return
    await msg.answer("Интервью завершено. Проверьте черновик перед публикацией.")
    await _show_review(msg, sess["id"])


async def _show_review(msg: Message, session_id: int) -> None:
    rows = await db.review_items(session_id)
    live = [r for r in rows if not r["deleted"]]
    lines = ["Черновик экспертных пунктов:", ""]
    lines.extend(f"{r['ord']}. {r['text']}" for r in live)
    lines += [
        "",
        "/edit N новый текст — исправить",
        "/delete N — удалить",
        "/add текст — добавить",
        "/approve — подтвердить и запустить публикацию",
    ]
    await _send_long(msg, "\n".join(lines))


async def _draft_for_msg(msg: Message):
    sess = await _active_session_for_msg(msg)
    if not sess or sess["status"] != "draft_review":
        await msg.answer("Нет черновика на проверке. Сначала завершите интервью: /finish.")
        return None
    return sess


@dp.message(Command("review"))
async def on_review(msg: Message) -> None:
    sess = await _draft_for_msg(msg)
    if sess:
        await _show_review(msg, sess["id"])


@dp.message(Command("edit"))
async def on_edit(msg: Message, command: CommandObject) -> None:
    sess = await _draft_for_msg(msg)
    if not sess:
        return
    raw = (command.args or "").strip().split(maxsplit=1)
    if len(raw) != 2 or not raw[0].isdigit() or not raw[1].strip():
        await msg.answer("Формат: /edit N новый текст")
        return
    if not await db.update_review_item(sess["id"], int(raw[0]), raw[1]):
        await msg.answer("Пункт не найден.")
        return
    await _show_review(msg, sess["id"])


@dp.message(Command("delete"))
async def on_delete(msg: Message, command: CommandObject) -> None:
    sess = await _draft_for_msg(msg)
    if not sess:
        return
    raw = (command.args or "").strip()
    if not raw.isdigit() or not await db.delete_review_item(sess["id"], int(raw)):
        await msg.answer("Формат: /delete N; пункт должен существовать.")
        return
    await _show_review(msg, sess["id"])


@dp.message(Command("add"))
async def on_add(msg: Message, command: CommandObject) -> None:
    sess = await _draft_for_msg(msg)
    if not sess:
        return
    text = (command.args or "").strip()
    if not text:
        await msg.answer("Формат: /add текст пункта")
        return
    await db.add_review_item(sess["id"], text)
    await _show_review(msg, sess["id"])


@dp.message(Command("approve"))
async def on_approve(msg: Message) -> None:
    sess = await _draft_for_msg(msg)
    if not sess:
        return
    ok = await db.approve_review(sess["id"], sess["user_id"], msg.chat.id)
    if not ok:
        await msg.answer("Не удалось подтвердить: черновик пуст или уже закрыт.")
        return
    await msg.answer("Черновик подтверждён. Durable job поставлен в очередь.")


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
    user_id = _telegram_user_id(msg)
    if user_id is None:
        return
    access = await db.ensure_user_workspace(
        user_id, _telegram_username(msg), _telegram_full_name(msg)
    )
    if access["role"] not in {"owner", "admin"}:
        await msg.answer("/verbose доступен только администратору workspace.")
        return
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
        fresh = await _fresh_owned_session_for_msg(
            msg, sess["id"], {"active", "draft_review"}
        )
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
    return await _fresh_owned_session_for_msg(msg, session_id, {"active"})


async def _fresh_owned_session_for_msg(
    msg: Message, session_id: int, statuses: set[str]
):
    user_id = _telegram_user_id(msg)
    if user_id is None:
        return None
    sess = await db.get_session(session_id)
    if not sess:
        return None
    if sess["status"] not in statuses:
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
        await msg.answer("Спасибо! Проверьте черновик перед публикацией.")
        await _show_review(msg, fresh["id"])


@dp.message(F.text)
async def on_text(msg: Message) -> None:
    sess = await _active_session_for_msg(msg)
    if not sess:
        await msg.answer("Нет активной сессии. Наберите /start чтобы начать.")
        return
    if sess["status"] == "draft_review":
        await msg.answer("Интервью на проверке. /review, затем /approve.")
        return
    if sess["status"] != "active":
        await msg.answer("Интервью уже подтверждено и обрабатывается.")
        return
    await _expert_turn(msg, sess, msg.text)


@dp.message(F.voice)
async def on_voice(msg: Message) -> None:
    sess = await _active_session_for_msg(msg)
    if not sess:
        await msg.answer("Нет активной сессии. Наберите /start чтобы начать.")
        return
    if sess["status"] != "active":
        await msg.answer("Интервью закрыто для новых ответов. Используйте /review.")
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


async def main() -> None:
    worker_stop: asyncio.Event | None = None
    worker_task: asyncio.Task | None = None
    polling_task: asyncio.Task | None = None
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
        worker_stop = asyncio.Event()

        async def _notify(job, rows) -> None:
            if job["chat_id"] is not None:
                await bot.send_message(job["chat_id"], _recap(rows))

        worker_task = asyncio.create_task(
            worker.run(worker_stop, _notify), name="durable-postprocess-worker"
        )
        await bot.set_my_commands([
            BotCommand(command="review", description="Показать черновик"),
            BotCommand(command="approve", description="Подтвердить черновик"),
            BotCommand(command="start", description="Начать интервью (/start тема)"),
            BotCommand(command="plan", description="Показать план и прогресс интервью"),
            BotCommand(command="finish", description="Завершить и обработать"),
            BotCommand(command="verbose", description="Показать/скрыть внутреннюю работу (отладка)"),
            BotCommand(command="reset", description="Удалить активную сессию, начать с нуля"),
        ])
        polling_task = asyncio.create_task(dp.start_polling(bot), name="telegram-polling")
        done, _ = await asyncio.wait(
            {polling_task, worker_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if worker_task in done:
            error = worker_task.exception()
            if error is not None:
                raise error
            raise RuntimeError("durable postprocess worker stopped unexpectedly")
        await polling_task
    finally:
        if polling_task is not None and not polling_task.done():
            polling_task.cancel()
            await asyncio.gather(polling_task, return_exceptions=True)
        if worker_stop is not None:
            worker_stop.set()
        if worker_task is not None:
            await asyncio.gather(worker_task, return_exceptions=True)
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
