"""Build the compact STATE block injected each turn.

Hard budget: keep it thin. One line per plan subtopic + a trimmed topic-summary
excerpt (open questions / contradictions first). NOT a full checklist or the
whole base — that provokes questionnaire mode over deep interview, and the agent
can pull the rest just-in-time via search_knowledge.
"""
import json

from . import config, db, embed

# ~800-token budget for the summary excerpt. Russian runs ~2 chars/token, so
# ~1600 chars. Full summary in the DB can grow unbounded; this is only the tail
# that rides in the prompt.
_SUMMARY_CAP = 1600
_RAG_ITEMS = 4        # cross-session facts pulled per turn
_RAG_ITEM_CHARS = 160  # each trimmed to this


def _item_gist(payload) -> str:
    if isinstance(payload, str):
        payload = json.loads(payload)
    text = payload.get("statement") or payload.get("answer") \
        or payload.get("definition") or json.dumps(payload, ensure_ascii=False)
    q = payload.get("qualifiers")
    if q:
        text = f"{text} ({q})"
    return text[:_RAG_ITEM_CHARS]


async def _auto_rag(session_id: int, topic: str, last_expert_text: str) -> list[str]:
    """Vector-search the base by the expert's last turn and surface a few
    canonical facts from past sessions. Scales to hundreds of sessions where the
    flat summary can't, and guarantees the cross-check context is present without
    the model having to remember to call search_knowledge.

    Facts split by author: a returning expert's OWN past statements must not be
    presented as 'чужие' (the agent would ask him to cross-check himself) —
    those are 'already told, don't re-ask' context instead."""
    if not embed.enabled() or not last_expert_text.strip():
        return []
    vec = await embed.embed(last_expert_text, query=True)
    if vec is None:
        return []
    rows = await db.search_canonical(
        topic, vec, limit=_RAG_ITEMS, exclude_session=session_id
    )
    if not rows:
        return []
    sess = await db.get_session(session_id)
    me = sess["expert_name"] if sess else ""
    mine = [r for r in rows if r["expert_name"] == me]
    others = [r for r in rows if r["expert_name"] != me]
    out: list[str] = []
    if mine:
        out += [
            "",
            "Из базы — ЭТОТ ЖЕ эксперт уже говорил в прошлых сессиях",
            "(не переспрашивай то же самое; ссылайся и копай глубже/новое):",
        ]
        out += [f"  - {_item_gist(r['payload'])}" for r in mine]
    if others:
        out += [
            "",
            "Из базы — ЧУЖИЕ утверждения других экспертов (для сверки/уточнения,",
            "не приписывай текущему эксперту):",
        ]
        out += [
            f"  - [эксперт {r['expert_name']}] {_item_gist(r['payload'])}"
            for r in others
        ]
    return out


async def build(
    session_id: int, topic: str, tokens_used: int = 0,
    last_expert_text: str = "",
) -> str:
    lines: list[str] = ["=== STATE (служебное, эксперту не видно) ==="]

    items = await db.plan_items(session_id)
    if items:
        # The tool rounds aren't replayed into the next turn's history, so the
        # model doesn't remember it already sent (or got refused) update_plan.
        # Inject that memory here (the per-turn dynamic slot): the call count
        # makes the ritual visible to the model itself.
        n_plan = (await db.tool_call_names(session_id)).count("update_plan")
        lines.append(
            f"План уже задан (ниже); update_plan вызван {n_plan} раз(а) за "
            "сессию. В этом ходе НЕ вызывай update_plan — исключение только "
            "одно: добавляешь или удаляешь подтему. Пересылка того же плана "
            "будет отклонена и просто сожжёт токены:"
        )
        for r in items:
            mark = "x" if r["status"] == "covered" else " "
            lines.append(f"  [{mark}] {r['subtopic']}")
    else:
        lines.append("План ещё не задан — после разогрева вызови update_plan.")

    summary = await db.topic_summary(topic)
    if summary:
        trimmed = summary[:_SUMMARY_CAP]
        if len(summary) > _SUMMARY_CAP:
            trimmed += " …[обрезано — детали через search_knowledge]"
        lines.append("")
        lines.append("Сводка по теме из прошлых сессий (строй план вокруг")
        lines.append("открытых вопросов и противоречий; детали — search_knowledge):")
        lines.append(trimmed)

    lines.extend(await _auto_rag(session_id, topic, last_expert_text))

    # Session token budget: soft cap nudges wrap-up, hard cap forces end.
    if config.HARD_CAP_TOKENS and tokens_used >= config.HARD_CAP_TOKENS:
        lines.append("")
        lines.append(
            "БЮДЖЕТ ИСЧЕРПАН. Заверши сейчас: вызови end_session с резюме "
            "собранного, новых вопросов не задавай."
        )
    elif config.SOFT_CAP_TOKENS and tokens_used >= config.SOFT_CAP_TOKENS:
        lines.append("")
        lines.append(
            "Бюджет на исходе — закругляйся: покрой самое важное из оставшегося "
            "и завершай."
        )

    return "\n".join(lines)
