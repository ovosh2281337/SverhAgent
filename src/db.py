"""Thin asyncpg wrapper. One pool for the process."""
import asyncio
import json
from typing import Any, Optional

import asyncpg

from . import config

_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()  # two concurrent first calls must not create two pools


async def _init_conn(conn: asyncpg.Connection) -> None:
    # Register the pgvector codec so Python lists bind to vector params. Harmless
    # to skip if the extension is absent — we only pass vectors when embeddings
    # are enabled, and those paths require pgvector anyway.
    try:
        from pgvector.asyncpg import register_vector

        await register_vector(conn)
    except Exception:
        pass


async def pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                _pool = await asyncpg.create_pool(
                    config.DATABASE_URL, min_size=1, max_size=5, init=_init_conn
                )
    return _pool


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# --- sessions ---------------------------------------------------------------

async def active_session(expert_name: str) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow(
        "SELECT * FROM sessions WHERE expert_name=$1 AND status='active' "
        "ORDER BY id DESC LIMIT 1",
        expert_name,
    )


async def get_session(session_id: int) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow("SELECT * FROM sessions WHERE id=$1", session_id)


async def create_session(expert_name: str, topic: str) -> asyncpg.Record:
    p = await pool()
    return await p.fetchrow(
        "INSERT INTO sessions (expert_name, topic, prompt_version) "
        "VALUES ($1, $2, $3) RETURNING *",
        expert_name, topic, config.PROMPT_VERSION,
    )


async def delete_session(session_id: int) -> None:
    """Hard-delete a session and its messages/plan (cascade). Used by /reset."""
    p = await pool()
    await p.execute("DELETE FROM sessions WHERE id=$1", session_id)


async def finish_session(session_id: int) -> None:
    p = await pool()
    await p.execute(
        "UPDATE sessions SET status='finished', finished_at=now() "
        "WHERE id=$1 AND status='active'",
        session_id,
    )


async def claim_for_extraction(session_id: int) -> bool:
    """Atomic CAS finished->extracting. True only for the caller that won it —
    makes double extraction impossible even on a session finished twice."""
    p = await pool()
    row = await p.fetchrow(
        "UPDATE sessions SET status='extracting' "
        "WHERE id=$1 AND status='finished' RETURNING id",
        session_id,
    )
    return row is not None


async def wipe_extracted(session_id: int) -> None:
    """Remove a session's extracted items — cleanup of a crashed extraction."""
    p = await pool()
    await p.execute(
        "DELETE FROM extracted_items WHERE session_id=$1", session_id
    )


async def revert_extraction(session_id: int) -> None:
    """extracting -> finished, so a crashed extraction can be retried."""
    p = await pool()
    await p.execute(
        "UPDATE sessions SET status='finished' "
        "WHERE id=$1 AND status='extracting'",
        session_id,
    )


async def mark_extracted(session_id: int) -> None:
    p = await pool()
    await p.execute(
        "UPDATE sessions SET status='extracted' WHERE id=$1", session_id
    )


async def add_tokens(session_id: int, n: int) -> int:
    p = await pool()
    return await p.fetchval(
        "UPDATE sessions SET tokens_used = tokens_used + $2 "
        "WHERE id=$1 RETURNING tokens_used",
        session_id, n,
    )


async def session_tokens(session_id: int) -> int:
    p = await pool()
    return await p.fetchval(
        "SELECT tokens_used FROM sessions WHERE id=$1", session_id
    ) or 0


# --- messages ---------------------------------------------------------------

async def add_message(
    session_id: int, role: str, content: str, tool_calls: Any = None
) -> int:
    p = await pool()
    return await p.fetchval(
        "INSERT INTO messages (session_id, role, content, tool_calls) "
        "VALUES ($1, $2, $3, $4) RETURNING id",
        session_id, role, content,
        json.dumps(tool_calls) if tool_calls is not None else None,
    )


async def tool_call_names(session_id: int) -> list[str]:
    """Flat list of tool names called so far in a session (persisted on
    assistant messages). Tool rounds are NOT replayed into the next turn's
    history, so without this the model has no memory of its own tool usage —
    the root cause of the resend-the-same-plan habit (see state.build)."""
    p = await pool()
    rows = await p.fetch(
        "SELECT tool_calls FROM messages "
        "WHERE session_id=$1 AND tool_calls IS NOT NULL",
        session_id,
    )
    names: list[str] = []
    for r in rows:
        calls = r["tool_calls"]
        if isinstance(calls, str):
            calls = json.loads(calls)
        names.extend(c.get("name", "?") for c in calls)
    return names


async def history(session_id: int) -> list[asyncpg.Record]:
    """User/assistant turns in order — for rebuilding the API message list."""
    p = await pool()
    return await p.fetch(
        "SELECT id, role, content FROM messages "
        "WHERE session_id=$1 AND role IN ('user','assistant') ORDER BY id",
        session_id,
    )


async def transcript(session_id: int) -> list[asyncpg.Record]:
    p = await pool()
    return await p.fetch(
        "SELECT id, role, content FROM messages WHERE session_id=$1 ORDER BY id",
        session_id,
    )


async def last_expert_message(session_id: int) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow(
        "SELECT id, content FROM messages WHERE session_id=$1 AND role='user' "
        "ORDER BY id DESC LIMIT 1",
        session_id,
    )


# --- plan -------------------------------------------------------------------

async def plan_items(session_id: int) -> list[asyncpg.Record]:
    p = await pool()
    return await p.fetch(
        "SELECT subtopic, status, ord FROM plan_items "
        "WHERE session_id=$1 ORDER BY ord, id",
        session_id,
    )


async def set_plan(session_id: int, subtopics: list[str]) -> None:
    """Replace the plan. Keeps 'covered' status for subtopics that survive."""
    p = await pool()
    async with p.acquire() as conn, conn.transaction():
        covered = {
            r["subtopic"]
            for r in await conn.fetch(
                "SELECT subtopic FROM plan_items "
                "WHERE session_id=$1 AND status='covered'",
                session_id,
            )
        }
        await conn.execute("DELETE FROM plan_items WHERE session_id=$1", session_id)
        for ord_, sub in enumerate(subtopics):
            await conn.execute(
                "INSERT INTO plan_items (session_id, subtopic, status, ord) "
                "VALUES ($1, $2, $3, $4)",
                session_id, sub, "covered" if sub in covered else "open", ord_,
            )


async def mark_covered(session_id: int, subtopic: str) -> None:
    p = await pool()
    await p.execute(
        "UPDATE plan_items SET status='covered' "
        "WHERE session_id=$1 AND subtopic=$2",
        session_id, subtopic,
    )


# --- topic summary ----------------------------------------------------------

async def topic_summary(topic: str) -> Optional[str]:
    p = await pool()
    return await p.fetchval(
        "SELECT summary FROM topic_summaries WHERE topic=$1", topic
    )


async def upsert_topic_summary(topic: str, summary: str) -> None:
    p = await pool()
    await p.execute(
        "INSERT INTO topic_summaries (topic, summary, prompt_version, generated_at) "
        "VALUES ($1, $2, $3, now()) "
        "ON CONFLICT (topic) DO UPDATE SET summary=EXCLUDED.summary, "
        "prompt_version=EXCLUDED.prompt_version, generated_at=now()",
        topic, summary, config.PROMPT_VERSION,
    )


# --- extracted items --------------------------------------------------------

async def add_extracted_item(
    session_id: int, type_: str, origin: str, payload: dict, quote: str,
    source_message_id: Optional[int], embedding: Optional[list[float]] = None,
    duplicate_of: Optional[int] = None,
) -> int:
    p = await pool()
    return await p.fetchval(
        "INSERT INTO extracted_items "
        "(session_id, type, origin, payload, quote, source_message_id, "
        " embedding, duplicate_of, prompt_version, embed_version) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id",
        session_id, type_, origin, json.dumps(payload, ensure_ascii=False),
        quote, source_message_id, embedding, duplicate_of, config.PROMPT_VERSION,
        config.EMBED_TEXT_VERSION,
    )


async def bump_confirmation(item_id: int) -> None:
    p = await pool()
    await p.execute(
        "UPDATE extracted_items SET confirmation_count = confirmation_count + 1 "
        "WHERE id=$1",
        item_id,
    )


async def nearest_canonical(
    topic: str, embedding: list[float]
) -> Optional[asyncpg.Record]:
    """Closest canonical fact of the same topic by cosine distance."""
    p = await pool()
    return await p.fetchrow(
        "SELECT e.id, e.payload, e.quote, "
        "       e.embedding <=> $2 AS dist "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE s.topic=$1 AND e.duplicate_of IS NULL AND e.embedding IS NOT NULL "
        "ORDER BY dist LIMIT 1",
        topic, embedding,
    )


async def search_canonical(
    topic: str, embedding: list[float], limit: int = 5,
    exclude_session: Optional[int] = None,
) -> list[asyncpg.Record]:
    """Top-k canonical facts of the topic nearest a query embedding (JIT context
    for the search_knowledge tool and the auto-RAG STATE block). exclude_session
    drops the current session's own items so auto-RAG shows only OTHER experts."""
    p = await pool()
    return await p.fetch(
        "SELECT e.type, e.payload, e.quote, e.origin, e.confirmation_count, "
        "       s.expert_name, e.embedding <=> $2 AS dist "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE s.topic=$1 AND e.duplicate_of IS NULL AND e.embedding IS NOT NULL "
        "  AND ($4::int IS NULL OR e.session_id <> $4) "
        "ORDER BY dist LIMIT $3",
        topic, embedding, limit, exclude_session,
    )


async def extracted_for_session(session_id: int) -> list[asyncpg.Record]:
    """Items collected in one session — for the post-finish recap to the expert."""
    p = await pool()
    return await p.fetch(
        "SELECT type, origin, payload, quote, duplicate_of "
        "FROM extracted_items WHERE session_id=$1 ORDER BY id",
        session_id,
    )


async def extracted_count(session_id: int) -> int:
    p = await pool()
    return await p.fetchval(
        "SELECT count(*) FROM extracted_items WHERE session_id=$1", session_id
    )


async def canonical_for_topic(topic: str) -> list[asyncpg.Record]:
    """Canonical items only (duplicate_of IS NULL) — the summary reads these."""
    p = await pool()
    return await p.fetch(
        "SELECT e.type, e.origin, e.payload, e.quote, e.confirmation_count, "
        "       s.expert_name "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE s.topic=$1 AND e.duplicate_of IS NULL ORDER BY e.id",
        topic,
    )


async def all_for_topic(topic: str) -> list[asyncpg.Record]:
    """Every item incl. duplicates — for inspection (scripts/view)."""
    p = await pool()
    return await p.fetch(
        "SELECT e.id, e.type, e.origin, e.payload, e.quote, "
        "       e.confirmation_count, e.duplicate_of, s.expert_name "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE s.topic=$1 ORDER BY e.id",
        topic,
    )


# --- question evals ---------------------------------------------------------

async def evaluated_message_ids(session_id: int) -> set[int]:
    p = await pool()
    rows = await p.fetch(
        "SELECT qe.message_id FROM question_evals qe "
        "JOIN messages m ON m.id = qe.message_id WHERE m.session_id=$1",
        session_id,
    )
    return {r["message_id"] for r in rows}


async def add_question_eval(
    message_id: int, expert_answer_message_id: Optional[int], verdict: dict
) -> None:
    p = await pool()
    await p.execute(
        "INSERT INTO question_evals "
        "(message_id, expert_answer_message_id, verdict, prompt_version) "
        "VALUES ($1, $2, $3, $4)",
        message_id, expert_answer_message_id,
        json.dumps(verdict, ensure_ascii=False), config.PROMPT_VERSION,
    )
