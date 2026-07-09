"""Agent tool definitions + applying tool calls to the DB.

Plan/coverage tools are advisory side-effects — the wrapper is robust to the
model forgetting to call them (coverage is a hint map, not hard state).
search_knowledge is just-in-time retrieval: the agent pulls relevant facts from
the base only when the expert touches something familiar, instead of preloading
the whole base into the prompt. This same tool becomes the RAG retrieval
endpoint later almost unchanged.
"""
import json
import re
from typing import Any

from . import db, embed, websearch


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    # strict=True so the SDK's streaming helper can auto-parse tool args; it
    # requires additionalProperties:false and every property in `required`.
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# OpenAI function-calling schema.
TOOLS: list[dict[str, Any]] = [
    _fn(
        "update_plan",
        "Создать или полностью заменить план интервью списком подтем. "
        "Вызывай после разогрева и когда эксперт раскрыл новую область.",
        {
            "subtopics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Подтемы плана по порядку.",
            }
        },
        ["subtopics"],
    ),
    _fn(
        "mark_covered",
        "Пометить подтему плана покрытой (точное название подтемы).",
        {"subtopic": {"type": "string"}},
        ["subtopic"],
    ),
    _fn(
        "search_knowledge",
        "Поиск по накопленной базе знаний темы. Используй, когда эксперт "
        "затронул что-то, что могли уже обсуждать другие эксперты — чтобы "
        "спросить сверку («ваш коллега утверждал X, вы согласны?») или не "
        "повторять покрытое. Возвращает ближайшие факты с цитатами.",
        {"query": {"type": "string", "description": "О чём ищем (тезис/тема)."}},
        ["query"],
    ),
    _fn(
        "web_search",
        "Проверить факт/термин/железо в вебе ПРЕЖДЕ чем строить на нём вопрос "
        "(существует ли модель, какие у неё характеристики, что значит термин "
        "эксперта). Возвращает сводку и выдержки страниц. Результат — фон для "
        "умного вопроса, не пересказывай его эксперту.",
        {"query": {"type": "string", "description": "Поисковый запрос."}},
        ["query"],
    ),
    _fn(
        "web_fetch",
        "Вытащить полный текст страницы по URL из результатов web_search, когда "
        "нужны детали, которых нет в выдержке.",
        {"url": {"type": "string", "description": "URL страницы."}},
        ["url"],
    ),
    _fn(
        "end_session",
        "Завершить сессию. summary — резюме «вот что я записал» для валидации "
        "экспертом. Текст резюме будет показан эксперту как последняя реплика.",
        {"summary": {"type": "string"}},
        ["summary"],
    ),
]

# Tools that require an optional backend; hidden from the model when off so it
# doesn't waste a reasoning round on a tool that can only answer "недоступно".
_GATED = {
    "search_knowledge": embed.enabled,
    "web_search": websearch.enabled,
    "web_fetch": websearch.enabled,
}


def active_tools() -> list[dict[str, Any]]:
    """Tools offered to the model this run. Backend-gated tools are dropped when
    their backend is off (embeddings for search_knowledge, Tavily key for
    web_*): otherwise the model burns a reasoning round on a tool that can only
    answer 'недоступно'. Each lights up automatically once its env is set."""
    return [
        t for t in TOOLS
        if _GATED.get(t["function"]["name"], lambda: True)()
    ]


def _plan_words(subs: list[str]) -> set[str]:
    text = " ".join(subs).lower()
    return {w for w in re.split(r"[^\wа-яё]+", text) if len(w) > 2}


def _plan_similar(a: list[str], b: list[str], thresh: float = 0.7) -> bool:
    """Jaccard over the whole plan's word set. Catches reworded resends (same
    subtopics, shuffled words) that an exact match misses."""
    wa, wb = _plan_words(a), _plan_words(b)
    if not wa or not wb:
        return a == b
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union >= thresh


class SessionEnd(Exception):
    """Raised out of tool application to signal the loop the session is over."""

    def __init__(self, summary: str):
        self.summary = summary


async def _search_knowledge(topic: str, query: str) -> str:
    if not embed.enabled():
        return "(поиск по базе недоступен: embeddings не настроены)"
    vec = await embed.embed(query or "", query=True)
    if vec is None:
        return "(поиск по базе недоступен)"
    rows = await db.search_canonical(topic, vec, limit=5)
    if not rows:
        return "В базе по этой теме пока ничего похожего."
    out = []
    for r in rows:
        payload = r["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        origin = "гипотеза" if r["origin"] == "confirmed_hypothesis" else "факт"
        quote = (r["quote"] or "").strip()
        if len(quote) > 260:
            quote = quote[:260] + "…"
        out.append(
            f"- [{r['type']}/{origin}, support={r['support_mode']}, "
            f"подтверждений={r['confirmation_count']}, "
            f"эксперт={r['expert_name']}] "
            f"{json.dumps(payload, ensure_ascii=False)}"
            + (f" | expert_span: {quote}" if quote else "")
        )
    return "Найдено в базе:\n" + "\n".join(out)


async def apply(session_id: int, topic: str, name: str, args: dict) -> str:
    """Apply one tool call; return a short result string for the tool_result."""
    if name == "update_plan":
        subs = [s.strip() for s in args.get("subtopics", []) if s.strip()]
        current = [r["subtopic"] for r in await db.plan_items(session_id)]
        if current and _plan_similar(subs, current):
            # The model re-sent essentially the same plan (often reworded). Skip
            # the rewrite: protects covered-status and kills the per-turn churn
            # that inflated session 8 to 159k tokens.
            return ("план почти не изменился — не пересылай его без реально новой "
                    "или удалённой подтемы")
        await db.set_plan(session_id, subs)
        return f"план обновлён: {len(subs)} подтем"
    if name == "mark_covered":
        await db.mark_covered(session_id, args.get("subtopic", ""))
        return "ок, помечено"
    if name == "search_knowledge":
        return await _search_knowledge(topic, args.get("query", ""))
    if name == "web_search":
        return await websearch.search(args.get("query", ""))
    if name == "web_fetch":
        return await websearch.fetch(args.get("url", ""))
    if name == "end_session":
        raise SessionEnd(args.get("summary", ""))
    return f"неизвестный тул: {name}"
