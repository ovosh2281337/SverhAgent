"""One interview turn end to end: build context, run the loop, persist output."""
import logging
from typing import Any

from . import config, db, llm, prompts, state, tools

log = logging.getLogger("agent")


async def _build_messages(
    session_id: int, topic: str, tokens_used: int
) -> tuple[list[dict[str, Any]], str]:
    """Returns (messages, state_block). The state block is handed back separately
    so callers can surface it (verbose/debug mode shows the exact context the
    model saw), not just embed it into the prompt."""
    rows = await db.history(session_id)
    # System prompt is stable; the dynamic STATE block rides on the tail of the
    # last user turn (keeps the system message stable, injects fresh state).
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.DIALOG_SYSTEM}
    ]
    turns = [{"role": r["role"], "content": r["content"]} for r in rows]
    last_expert_text = next(
        (t["content"] for t in reversed(turns) if t["role"] == "user"), ""
    )
    state_block = await state.build(
        session_id, topic, tokens_used, last_expert_text=last_expert_text
    )
    for msg in reversed(turns):
        if msg["role"] == "user":
            msg["content"] = f"{msg['content']}\n\n{state_block}"
            break
    messages.extend(turns)
    return messages, state_block


async def run_turn(session_id: int, topic: str) -> tuple[str, bool, dict[str, Any]]:
    """Produce the next agent message.

    Returns (text, session_finished, trace). `trace` exposes what happened this
    turn for observability/verbose mode: the tool calls made (name + args), the
    STATE context fed to the model, and the token spend."""
    tokens_used = await db.session_tokens(session_id)
    messages, state_block = await _build_messages(session_id, topic, tokens_used)

    ended = {"summary": None}
    used: list[dict[str, Any]] = []  # tool calls this turn, for persistence/observability

    async def apply_tool(name: str, args: dict) -> str:
        used.append({"name": name, "args": args})
        try:
            return await tools.apply(session_id, topic, name, args)
        except tools.SessionEnd as e:
            ended["summary"] = e.summary
            return "сессия помечена завершённой"
        except Exception:
            # A failed tool must not kill the turn: normalize the error into a
            # tool result so the model can route around it (ask without the
            # tool, or try another one) instead of the expert seeing a crash.
            log.exception("tool failed: %s", name)
            return f"тул {name} упал с внутренней ошибкой — продолжай без него"

    rounds: list[dict[str, Any]] = []
    text, spent = await llm.dialogue(
        messages, tools.active_tools(), apply_tool, rounds_out=rounds
    )
    total = await db.add_tokens(session_id, spent)
    if used:
        log.info("session=%s tools=%s spent=%s total=%s",
                 session_id, [u["name"] for u in used], spent, total)
    tool_calls = used or None
    trace = {
        "tools": used, "state": state_block, "spent": spent,
        "total": total, "rounds": rounds,
    }

    if ended["summary"] is not None:
        final = ended["summary"] or text or "Спасибо, на этом закончим."
        await db.add_message(session_id, "assistant", final, tool_calls=tool_calls)
        await db.finish_session(session_id)
        return final, True, trace

    await db.add_message(session_id, "assistant", text, tool_calls=tool_calls)

    # Hard-cap safety net: if the model blew past the budget without ending, stop
    # the session ourselves (the STATE instruction is the softer first line).
    if config.HARD_CAP_TOKENS and total >= config.HARD_CAP_TOKENS:
        await db.finish_session(session_id)
        return text, True, trace

    return text, False, trace
