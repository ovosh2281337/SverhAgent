"""Thin asyncpg wrapper. One pool for the process."""
import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import asyncpg

from . import config

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


async def migrate() -> list[str]:
    """Apply every migrations/*.sql not yet recorded, in filename order.

    Idempotent and safe to run on every boot: a schema_migrations ledger skips
    already-applied files, and the SQL itself uses IF NOT EXISTS. This removes
    the classic footgun where docker's initdb only runs on an empty volume, so
    a migration added later never lands on an existing database. Returns the
    filenames applied this call (empty when already up to date)."""
    p = await pool()
    await p.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    done = {r["filename"] for r in await p.fetch("SELECT filename FROM schema_migrations")}
    applied: list[str] = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        if path.name in done:
            continue
        async with p.acquire() as conn:
            async with conn.transaction():
                await conn.execute(path.read_text(encoding="utf-8"))
                await conn.execute(
                    "INSERT INTO schema_migrations(filename) VALUES ($1)", path.name
                )
        applied.append(path.name)
    return applied

_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()  # two concurrent first calls must not create two pools


class ActiveSessionExistsError(RuntimeError):
    """Raised when Telegram user already has an active interview session."""


async def _init_conn(conn: asyncpg.Connection) -> None:
    # The schema always uses vector(640), even when the embedding service is
    # intentionally disabled. Install/register the type before migrations so a
    # fresh database and every pooled connection have the same codec contract.
    try:
        from pgvector.asyncpg import register_vector

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
    except Exception as exc:
        raise RuntimeError(
            "pgvector extension/asyncpg codec initialization failed"
        ) from exc


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


async def vector_health_check() -> None:
    """Fail fast if Python vectors cannot be bound through the pool codec."""
    p = await pool()
    dims = await p.fetchval(
        "SELECT vector_dims($1::vector)", [0.0] * config.EMBED_DIM
    )
    if dims != config.EMBED_DIM:
        raise RuntimeError(
            f"pgvector codec dimension check failed: {dims} != {config.EMBED_DIM}"
        )


# --- sessions ---------------------------------------------------------------

async def active_session_for_user(telegram_user_id: int) -> Optional[asyncpg.Record]:
    if telegram_user_id <= 0:
        raise ValueError("telegram_user_id must be positive")
    p = await pool()
    return await p.fetchrow(
        "SELECT * FROM sessions WHERE telegram_user_id=$1 AND status='active' "
        "ORDER BY id DESC LIMIT 1",
        telegram_user_id,
    )


async def get_session(session_id: int) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow("SELECT * FROM sessions WHERE id=$1", session_id)


async def create_session(
    expert_name: str,
    topic: str,
    *,
    telegram_user_id: Optional[int] = None,
    telegram_username: Optional[str] = None,
    telegram_full_name: Optional[str] = None,
) -> asyncpg.Record:
    if telegram_user_id is not None and telegram_user_id <= 0:
        raise ValueError("telegram_user_id must be positive")
    p = await pool()
    try:
        return await p.fetchrow(
            "INSERT INTO sessions "
            "(expert_name, topic, prompt_version, telegram_user_id, "
            " telegram_username, telegram_full_name) "
            "VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
            expert_name, topic, config.PROMPT_VERSION,
            telegram_user_id, telegram_username, telegram_full_name,
        )
    except asyncpg.UniqueViolationError as exc:
        if telegram_user_id is not None:
            raise ActiveSessionExistsError(
                f"telegram user {telegram_user_id} already has an active session"
            ) from exc
        raise


async def active_session_by_name_legacy(expert_name: str) -> Optional[asyncpg.Record]:
    """Legacy inspection helper only. The Telegram bot must not use names as keys."""
    p = await pool()
    return await p.fetchrow(
        "SELECT * FROM sessions WHERE expert_name=$1 AND status='active' "
        "ORDER BY id DESC LIMIT 1",
        expert_name,
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


async def claim_for_regrounding(session_id: int) -> bool:
    """Claim an extracted session that has only legacy knowledge. Existing
    legacy rows stay in place and remain unpublished throughout the new pass."""
    p = await pool()
    row = await p.fetchrow(
        "UPDATE sessions s SET status='extracting' "
        "WHERE s.id=$1 AND s.status='extracted' "
        "  AND EXISTS (SELECT 1 FROM extracted_items old "
        "              WHERE old.session_id=s.id "
        "                AND old.grounding_status='legacy') "
        "  AND NOT EXISTS (SELECT 1 FROM extracted_items current "
        "                  WHERE current.session_id=s.id "
        "                    AND current.grounding_version=$2 "
        "                    AND current.grounding_status<>'legacy') "
        "RETURNING s.id",
        session_id, config.GROUNDING_VERSION,
    )
    return row is not None


async def wipe_extracted(session_id: int) -> None:
    """Remove a session's extracted items — cleanup of a crashed extraction."""
    p = await pool()
    await p.execute(
        "DELETE FROM extracted_items WHERE session_id=$1", session_id
    )


async def wipe_grounding_version(session_id: int, grounding_version: str) -> None:
    """Rollback only rows created by the failed grounding pass. Legacy audit
    rows and rejection diagnostics are intentionally preserved."""
    p = await pool()
    await p.execute(
        "DELETE FROM extracted_items "
        "WHERE session_id=$1 AND grounding_version=$2 "
        "  AND grounding_status<>'legacy'",
        session_id, grounding_version,
    )


async def revert_extraction(session_id: int) -> None:
    """extracting -> finished, so a crashed extraction can be retried."""
    p = await pool()
    await p.execute(
        "UPDATE sessions SET status='finished' "
        "WHERE id=$1 AND status='extracting'",
        session_id,
    )


async def restore_extraction_status(session_id: int, status: str) -> None:
    if status not in {"finished", "extracted"}:
        raise ValueError(f"invalid extraction fallback status: {status}")
    p = await pool()
    await p.execute(
        "UPDATE sessions SET status=$2 WHERE id=$1 AND status='extracting'",
        session_id, status,
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
    *,
    session_id: int,
    type_: str,
    origin: str,
    payload: dict,
    primary: Any,
    provenance: list[Any] | tuple[Any, ...],
    support_mode: str,
    grounding_status: str,
    grounding_details: dict,
    embedding: Optional[list[float]] = None,
    duplicate_of: Optional[int] = None,
) -> int:
    """Atomically insert an item and every provenance span.

    The database's deferred constraint trigger validates the complete evidence
    graph at commit. A verified duplicate also refreshes confirmation_count in
    this same transaction, so a failed span insert cannot inflate reliability.
    """
    p = await pool()
    async with p.acquire() as conn:
        async with conn.transaction():
            item_id = await conn.fetchval(
                "INSERT INTO extracted_items "
                "(session_id, type, origin, payload, quote, source_message_id, "
                " embedding, duplicate_of, prompt_version, embed_version, "
                " support_mode, grounding_status, grounding_version, "
                " grounding_details) "
                "SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, "
                "       $11, $12, $13, $14 "
                "FROM messages m "
                "WHERE m.id=$6 AND m.session_id=$1 AND m.role='user' "
                "  AND substring(m.content FROM $15 + 1 FOR $16 - $15)=$5 "
                "RETURNING id",
                session_id, type_, origin,
                json.dumps(payload, ensure_ascii=False),
                primary.quote, primary.message_id, embedding, duplicate_of,
                config.PROMPT_VERSION, config.EMBED_TEXT_VERSION,
                support_mode, grounding_status, config.GROUNDING_VERSION,
                json.dumps(grounding_details, ensure_ascii=False),
                primary.start_char, primary.end_char,
            )
            if item_id is None:
                raise ValueError(
                    "primary support must be an exact expert span in this session"
                )

            for span in provenance:
                result = await conn.execute(
                    "INSERT INTO extracted_item_provenance "
                    "(item_id, message_id, kind, start_char, end_char, ord) "
                    "SELECT $1, $2, $3, $4, $5, $6 FROM messages m "
                    "JOIN extracted_items e ON e.id=$1 "
                    "WHERE m.id=$2 AND m.session_id=e.session_id "
                    "  AND substring(m.content FROM $4 + 1 FOR $5 - $4)=$7",
                    item_id, span.message_id, span.kind, span.start_char,
                    span.end_char, span.ord, span.quote,
                )
                if result != "INSERT 0 1":
                    raise ValueError(
                        f"provenance span {span.ord} does not match its message"
                    )
            return item_id


async def add_extraction_rejection(
    session_id: int,
    stage: str,
    reason: str,
    raw_item: object,
    attempted_repair: bool,
) -> int:
    safe_raw = raw_item if isinstance(raw_item, (dict, list)) else {
        "raw_repr": repr(raw_item)
    }
    p = await pool()
    return await p.fetchval(
        "INSERT INTO extraction_rejections "
        "(session_id, stage, reason, raw_item, attempted_repair, grounding_version) "
        "VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        session_id, stage, reason[:4000],
        json.dumps(safe_raw, ensure_ascii=False), attempted_repair,
        config.GROUNDING_VERSION,
    )


async def nearest_canonical(
    topic: str, embedding: list[float], current_session: int
) -> Optional[asyncpg.Record]:
    """Closest published fact, plus earlier rows of this extraction run.

    The current extracting session is included to collapse chunk-overlap
    duplicates. Rows from every other unfinished session stay invisible.
    """
    p = await pool()
    return await p.fetchrow(
        "SELECT e.id, e.payload, e.quote, e.support_mode, "
        "       e.embedding <=> $2 AS dist "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE s.topic=$1 AND (s.status='extracted' OR e.session_id=$5) "
        "  AND e.duplicate_of IS NULL AND e.embedding IS NOT NULL "
        "  AND e.embed_version=$3 "
        "  AND e.grounding_version=$4 "
        "  AND e.grounding_status='verified' "
        "ORDER BY dist LIMIT 1",
        topic, embedding, config.EMBED_TEXT_VERSION, config.GROUNDING_VERSION,
        current_session,
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
        "SELECT e.type, e.payload, e.quote, e.origin, e.support_mode, "
        "       e.confirmation_count, "
        "       s.expert_name, e.embedding <=> $2 AS dist "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE s.topic=$1 AND s.status='extracted' "
        "  AND e.duplicate_of IS NULL AND e.embedding IS NOT NULL "
        "  AND e.embed_version=$5 "
        "  AND e.grounding_version=$6 "
        "  AND e.grounding_status='verified' "
        "  AND ($4::bigint IS NULL OR e.session_id <> $4) "
        "  AND (e.embedding <=> $2) <= $7 "
        "ORDER BY dist LIMIT $3",
        topic, embedding, limit, exclude_session, config.EMBED_TEXT_VERSION,
        config.GROUNDING_VERSION, config.RAG_MAX_DISTANCE,
    )


async def extracted_for_session(session_id: int) -> list[asyncpg.Record]:
    """Verified current-version items collected in one session — for recap."""
    p = await pool()
    return await p.fetch(
        "SELECT type, origin, support_mode, payload, quote, duplicate_of "
        "FROM extracted_items "
        "WHERE session_id=$1 "
        "  AND grounding_status='verified' "
        "  AND grounding_version=$2 "
        "ORDER BY id",
        session_id, config.GROUNDING_VERSION,
    )


async def extracted_count(session_id: int) -> int:
    p = await pool()
    return await p.fetchval(
        "SELECT count(*) FROM extracted_items "
        "WHERE session_id=$1 "
        "  AND grounding_status='verified' "
        "  AND grounding_version=$2",
        session_id, config.GROUNDING_VERSION,
    )


async def canonical_for_topic(topic: str) -> list[asyncpg.Record]:
    """Only completed, current-version canonical items for topic summaries."""
    p = await pool()
    return await p.fetch(
        "SELECT e.type, e.origin, e.support_mode, e.payload, e.quote, "
        "       e.confirmation_count, s.expert_name, "
        "       COALESCE(( "
        "         SELECT jsonb_agg(jsonb_build_object( "
        "           'item_id', x.item_id, "
        "           'expert_name', x.expert_name, "
        "           'kind', x.kind, "
        "           'quote', x.quote "
        "         ) ORDER BY x.item_id, x.ord) "
        "         FROM ( "
        "           SELECT ev.id AS item_id, es.expert_name, p.kind, p.ord, "
        "                  substring(m.content FROM p.start_char + 1 "
        "                            FOR p.end_char - p.start_char) AS quote "
        "           FROM extracted_items ev "
        "           JOIN sessions es ON es.id=ev.session_id "
        "           JOIN extracted_item_provenance p ON p.item_id=ev.id "
        "           JOIN messages m ON m.id=p.message_id "
        "           WHERE (ev.id=e.id OR ev.duplicate_of=e.id) "
        "             AND ev.grounding_status='verified' "
        "             AND ev.grounding_version=$3 "
        "             AND p.kind IN ('user_support','confirmation') "
        "           ORDER BY ev.id, p.ord "
        "           LIMIT 4 "
        "         ) x "
        "       ), '[]'::jsonb) AS supports "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE s.topic=$1 AND s.status='extracted' "
        "  AND e.duplicate_of IS NULL "
        "  AND e.embed_version=$2 "
        "  AND e.grounding_version=$3 "
        "  AND e.grounding_status='verified' "
        "ORDER BY e.id",
        topic, config.EMBED_TEXT_VERSION, config.GROUNDING_VERSION,
    )


async def all_for_topic(topic: str) -> list[asyncpg.Record]:
    """Every item incl. duplicates — for inspection (scripts/view)."""
    p = await pool()
    return await p.fetch(
        "SELECT e.id, e.type, e.origin, e.support_mode, e.grounding_status, "
        "       e.grounding_version, e.grounding_details, e.payload, e.quote, "
        "       e.confirmation_count, e.duplicate_of, s.expert_name, "
        "       COALESCE(( "
        "         SELECT jsonb_agg(jsonb_build_object( "
        "           'message_id', p.message_id, "
        "           'kind', p.kind, "
        "           'quote', substring(m.content FROM p.start_char + 1 "
        "                             FOR p.end_char - p.start_char), "
        "           'ord', p.ord "
        "         ) ORDER BY p.ord) "
        "         FROM extracted_item_provenance p "
        "         JOIN messages m ON m.id=p.message_id "
        "         WHERE p.item_id=e.id "
        "       ), '[]'::jsonb) AS provenance "
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


if __name__ == "__main__":
    # python -m src.db migrate  — apply pending migrations, then exit.
    import sys

    async def _main() -> None:
        cmd = sys.argv[1] if len(sys.argv) > 1 else "migrate"
        if cmd != "migrate":
            print(f"unknown command: {cmd} (only 'migrate')")
            return
        applied = await migrate()
        print("applied: " + (", ".join(applied) if applied else "none (up to date)"))
        await close()

    asyncio.run(_main())
