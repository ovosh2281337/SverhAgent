"""Web search via Tavily — an LLM-oriented search API.

Unlike a 2023-era "scrape 10 blue links" tool, Tavily returns a synthesized
answer plus already-extracted page content, so the model gets usable context in
one call. The interviewer uses this mid-reasoning to sanity-check hardware/terms
before building a question on a possibly-wrong prior (see prompts.py v5).

Optional: without TAVILY_API_KEY every function returns a short "unavailable"
string and the web_search/web_fetch tools are hidden (tools.active_tools). All
errors are swallowed into that string — a flaky search must never abort a turn.

trust_env=True here (unlike llm.py/embed.py): Tavily is an external host, so the
system proxy is the right path out, not a localhost trap.
"""
import httpx

from . import config

_enabled = bool(config.TAVILY_API_KEY)
_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0)) if _enabled else None

_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"
_UNAVAIL = "(веб-поиск недоступен)"

# Per-process cache of successful lookups. Facts the agent checks (does model X
# exist, what MCU is on board Y) don't change mid-day, and the free Tavily tier
# is ~1000 req/mo — don't spend two requests on the same query.
_cache: dict[str, str] = {}
_CACHE_MAX = 256


def enabled() -> bool:
    return _enabled and _client is not None


async def search(query: str) -> str:
    """Synthesized answer + top results, trimmed for prompt use."""
    if not enabled() or not query.strip():
        return _UNAVAIL
    key = " ".join(query.lower().split())
    if key in _cache:
        return _cache[key]
    try:
        r = await _client.post(
            _SEARCH_URL,
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": query,
                "max_results": 5,
                "include_answer": True,
                "search_depth": "basic",
            },
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return _UNAVAIL

    parts: list[str] = []
    answer = (data.get("answer") or "").strip()
    if answer:
        parts.append(f"Сводка: {answer}")
    for res in data.get("results", []):
        title = (res.get("title") or "").strip()
        url = (res.get("url") or "").strip()
        content = (res.get("content") or "").strip()[:400]
        if content:
            parts.append(f"— {title} ({url}): {content}")
    if not parts:
        return "По запросу ничего не найдено."
    text = ("\n".join(parts))[:2500]
    if len(_cache) < _CACHE_MAX:
        _cache[key] = text
    return text


async def fetch(url: str) -> str:
    """Full extracted text of one page, trimmed. For drilling into a result."""
    if not enabled() or not url.strip():
        return _UNAVAIL
    try:
        r = await _client.post(
            _EXTRACT_URL,
            json={"api_key": config.TAVILY_API_KEY, "urls": [url]},
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return _UNAVAIL
    results = data.get("results") or []
    if not results:
        return "Страницу не удалось извлечь."
    return ((results[0].get("raw_content") or "").strip())[:3000] or "Пустая страница."
