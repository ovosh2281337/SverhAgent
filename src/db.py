"""Thin asyncpg wrapper. One pool for the process."""
import asyncio
import json
from contextlib import asynccontextmanager
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

async def ensure_user_workspace(
    telegram_user_id: int,
    telegram_username: Optional[str] = None,
    telegram_full_name: Optional[str] = None,
) -> asyncpg.Record:
    """Resolve Telegram identity to public collection or private workspace."""
    if telegram_user_id <= 0:
        raise ValueError("telegram_user_id must be positive")
    p = await pool()
    async with p.acquire() as conn, conn.transaction():
        user = await conn.fetchrow(
            "INSERT INTO users (telegram_user_id,telegram_username,telegram_full_name) "
            "VALUES ($1,$2,$3) ON CONFLICT (telegram_user_id) DO UPDATE SET "
            "telegram_username=COALESCE(EXCLUDED.telegram_username,users.telegram_username), "
            "telegram_full_name=COALESCE(EXCLUDED.telegram_full_name,users.telegram_full_name), "
            "updated_at=now() RETURNING *",
            telegram_user_id, telegram_username, telegram_full_name,
        )
        public = bool(config.PUBLIC_COLLECTION_SLUG)
        if public:
            workspace = await conn.fetchrow(
                "INSERT INTO workspaces(owner_user_id,name,slug) VALUES (NULL,$1,$2) "
                "ON CONFLICT (slug) DO UPDATE SET name=EXCLUDED.name RETURNING *",
                config.PUBLIC_COLLECTION_NAME, config.PUBLIC_COLLECTION_SLUG,
            )
            role = (
                "admin" if telegram_user_id in config.ADMIN_TELEGRAM_USER_IDS
                else "member"
            )
        else:
            workspace = await conn.fetchrow(
                "INSERT INTO workspaces(owner_user_id,name,slug) VALUES ($1,$2,$3) "
                "ON CONFLICT (owner_user_id) DO UPDATE SET name=EXCLUDED.name RETURNING *",
                user["id"],
                telegram_full_name or telegram_username or "Personal workspace",
                f"personal-{user['id']}",
            )
            role = "owner"
        await conn.execute(
            "INSERT INTO workspace_members(workspace_id,user_id,role) "
            "VALUES ($1,$2,$3) ON CONFLICT (workspace_id,user_id) "
            "DO UPDATE SET role=EXCLUDED.role",
            workspace["id"], user["id"], role,
        )
        return await conn.fetchrow(
            "SELECT $1::bigint AS user_id,$2::bigint AS workspace_id,$3::text AS role",
            user["id"], workspace["id"], role,
        )


async def ensure_legacy_workspace() -> asyncpg.Record:
    p = await pool()
    async with p.acquire() as conn, conn.transaction():
        user = await conn.fetchrow(
            "INSERT INTO users(telegram_full_name) VALUES ('CLI/legacy session') "
            "RETURNING *"
        )
        workspace = await conn.fetchrow(
            "INSERT INTO workspaces(owner_user_id,name,slug) "
            "VALUES ($1,'Legacy workspace',$2) ON CONFLICT (owner_user_id) "
            "DO UPDATE SET name=EXCLUDED.name RETURNING *",
            user["id"], f"personal-{user['id']}",
        )
        await conn.execute(
            "INSERT INTO workspace_members(workspace_id,user_id,role) "
            "VALUES ($1,$2,'owner') ON CONFLICT DO NOTHING",
            workspace["id"], user["id"],
        )
        return await conn.fetchrow(
            "SELECT $1::bigint AS user_id,$2::bigint AS workspace_id,'owner'::text AS role",
            user["id"], workspace["id"],
        )


async def resolve_topic(workspace_id: int, name: str) -> asyncpg.Record:
    clean = " ".join(name.strip().split())[:200] or "default"
    p = await pool()
    return await p.fetchrow(
        "INSERT INTO topics(workspace_id,name) VALUES ($1,$2) "
        "ON CONFLICT (workspace_id,name) DO UPDATE SET name=EXCLUDED.name RETURNING *",
        workspace_id, clean,
    )


async def get_topic(workspace_id: int, topic_id: int) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow(
        "SELECT * FROM topics WHERE workspace_id=$1 AND id=$2",
        workspace_id, topic_id,
    )


async def workspace_access_for_user(
    telegram_user_id: int, workspace_id: int
) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow(
        "SELECT u.id AS user_id,wm.workspace_id,wm.role FROM users u "
        "JOIN workspace_members wm ON wm.user_id=u.id "
        "WHERE u.telegram_user_id=$1 AND wm.workspace_id=$2",
        telegram_user_id, workspace_id,
    )

async def active_session_for_user(telegram_user_id: int) -> Optional[asyncpg.Record]:
    if telegram_user_id <= 0:
        raise ValueError("telegram_user_id must be positive")
    p = await pool()
    return await p.fetchrow(
        "SELECT s.* FROM sessions s JOIN users u ON u.id=s.user_id "
        "WHERE u.telegram_user_id=$1 AND s.status='active' "
        "ORDER BY s.id DESC LIMIT 1",
        telegram_user_id,
    )


async def open_session_for_user(telegram_user_id: int) -> Optional[asyncpg.Record]:
    if telegram_user_id <= 0:
        raise ValueError("telegram_user_id must be positive")
    p = await pool()
    return await p.fetchrow(
        "SELECT s.* FROM sessions s JOIN users u ON u.id=s.user_id "
        "WHERE u.telegram_user_id=$1 AND s.status IN ('active','draft_review') "
        "ORDER BY s.id DESC LIMIT 1",
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
    user_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    topic_id: Optional[int] = None,
) -> asyncpg.Record:
    if telegram_user_id is not None and telegram_user_id <= 0:
        raise ValueError("telegram_user_id must be positive")
    if user_id is None or workspace_id is None:
        access = (
            await ensure_user_workspace(
                telegram_user_id, telegram_username, telegram_full_name
            ) if telegram_user_id is not None else await ensure_legacy_workspace()
        )
        user_id, workspace_id = access["user_id"], access["workspace_id"]
    if topic_id is None:
        topic_row = await resolve_topic(workspace_id, topic)
        topic_id, topic = topic_row["id"], topic_row["name"]
    p = await pool()
    try:
        return await p.fetchrow(
            "INSERT INTO sessions "
            "(expert_name, topic, prompt_version, telegram_user_id, "
            " telegram_username, telegram_full_name,user_id,workspace_id,topic_id) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *",
            expert_name, topic, config.PROMPT_VERSION,
            telegram_user_id, telegram_username, telegram_full_name,
            user_id, workspace_id, topic_id,
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
    await begin_review(session_id)


async def begin_review(session_id: int) -> bool:
    """active -> draft_review; seed editable expert points."""
    p = await pool()
    async with p.acquire() as conn, conn.transaction():
        row = await conn.fetchrow(
            "UPDATE sessions SET status='draft_review',finished_at=now() "
            "WHERE id=$1 AND status='active' RETURNING id", session_id,
        )
        if row is None:
            return False
        await conn.execute(
            "UPDATE messages SET included_in_extraction=FALSE "
            "WHERE session_id=$1 AND role='user'", session_id,
        )
        await conn.execute(
            "INSERT INTO review_items(session_id,source_message_id,ord,text) "
            "SELECT $1,id,row_number() OVER (ORDER BY id)::int,content "
            "FROM messages WHERE session_id=$1 AND role='user' "
            "ON CONFLICT (session_id,ord) DO NOTHING", session_id,
        )
        return True


async def review_items(session_id: int) -> list[asyncpg.Record]:
    p = await pool()
    return await p.fetch(
        "SELECT id,ord,text,deleted FROM review_items WHERE session_id=$1 ORDER BY ord",
        session_id,
    )


async def update_review_item(session_id: int, ord_: int, text: str) -> bool:
    p = await pool()
    row = await p.fetchrow(
        "UPDATE review_items r SET text=$3,deleted=FALSE,updated_at=now() "
        "FROM sessions s WHERE r.session_id=s.id AND s.id=$1 "
        "AND s.status='draft_review' AND r.ord=$2 RETURNING r.id",
        session_id, ord_, text.strip(),
    )
    return row is not None


async def delete_review_item(session_id: int, ord_: int) -> bool:
    p = await pool()
    row = await p.fetchrow(
        "UPDATE review_items r SET deleted=TRUE,updated_at=now() "
        "FROM sessions s WHERE r.session_id=s.id AND s.id=$1 "
        "AND s.status='draft_review' AND r.ord=$2 RETURNING r.id",
        session_id, ord_,
    )
    return row is not None


async def add_review_item(session_id: int, text: str) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow(
        "INSERT INTO review_items(session_id,ord,text) "
        "SELECT s.id,COALESCE((SELECT max(ord)+1 FROM review_items "
        "WHERE session_id=s.id),1),$2 FROM sessions s "
        "WHERE s.id=$1 AND s.status='draft_review' RETURNING *",
        session_id, text.strip(),
    )


async def approve_review(session_id: int, user_id: int, chat_id: Optional[int]) -> bool:
    """Freeze reviewed points, finalize, enqueue exactly one durable job."""
    p = await pool()
    async with p.acquire() as conn, conn.transaction():
        sess = await conn.fetchrow(
            "SELECT * FROM sessions WHERE id=$1 AND user_id=$2 "
            "AND status='draft_review' FOR UPDATE", session_id, user_id,
        )
        if sess is None:
            return False
        count = await conn.fetchval(
            "SELECT count(*) FROM review_items WHERE session_id=$1 "
            "AND NOT deleted AND btrim(text)<>''", session_id,
        )
        if not count:
            return False
        await conn.execute(
            "INSERT INTO messages(session_id,role,content,tool_calls,included_in_extraction) "
            "SELECT $1,'user',text,jsonb_build_object('review_item_id',id,'approved',true),TRUE "
            "FROM review_items WHERE session_id=$1 AND NOT deleted "
            "AND btrim(text)<>'' ORDER BY ord", session_id,
        )
        await conn.execute(
            "UPDATE sessions SET status='finalized' WHERE id=$1", session_id,
        )
        await conn.execute(
            "INSERT INTO postprocess_jobs "
            "(workspace_id,session_id,topic_id,chat_id,extraction_version,"
            "prompt_version,model_version,idempotency_key) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8) "
            "ON CONFLICT (session_id) DO NOTHING",
            sess["workspace_id"], session_id, sess["topic_id"], chat_id,
            config.GROUNDING_VERSION, config.PROMPT_VERSION, config.EXTRACT_MODEL,
            f"postprocess:{session_id}:{config.GROUNDING_VERSION}",
        )
        return True


async def claim_for_extraction(session_id: int) -> bool:
    """Atomic CAS finished->extracting. True only for the caller that won it —
    makes double extraction impossible even on a session finished twice."""
    p = await pool()
    row = await p.fetchrow(
        "UPDATE sessions SET status='extracting' "
        "WHERE id=$1 AND status='finalized' RETURNING id",
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
        "UPDATE sessions SET status='finalized' "
        "WHERE id=$1 AND status='extracting'",
        session_id,
    )


async def restore_extraction_status(session_id: int, status: str) -> None:
    if status not in {"finalized", "extracted"}:
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


async def context_compaction(session_id: int) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow(
        "SELECT summary,through_message_id,prompt_version,updated_at "
        "FROM session_context_compactions WHERE session_id=$1",
        session_id,
    )


async def uncompacted_history_size(session_id: int, after_id: int) -> asyncpg.Record:
    p = await pool()
    return await p.fetchrow(
        "SELECT count(*)::bigint AS messages,"
        "COALESCE(sum(length(content)+128),0)::bigint AS chars "
        "FROM messages WHERE session_id=$1 AND id>$2 "
        "AND role IN ('user','assistant')",
        session_id, after_id,
    )


async def history_after(
    session_id: int, after_id: int, limit: int
) -> list[asyncpg.Record]:
    p = await pool()
    return await p.fetch(
        "SELECT id,role,content FROM messages WHERE session_id=$1 AND id>$2 "
        "AND role IN ('user','assistant') ORDER BY id LIMIT $3",
        session_id, after_id, limit,
    )


async def history_after_all(
    session_id: int, after_id: int
) -> list[asyncpg.Record]:
    p = await pool()
    return await p.fetch(
        "SELECT id,role,content FROM messages WHERE session_id=$1 AND id>$2 "
        "AND role IN ('user','assistant') ORDER BY id",
        session_id, after_id,
    )


async def recent_history_within_chars(
    session_id: int, after_id: int, char_budget: int
) -> list[asyncpg.Record]:
    """Newest complete turns fitting a token-derived character budget."""
    p = await pool()
    return await p.fetch(
        "WITH ranked AS (SELECT id,role,content,"
        "row_number() OVER (ORDER BY id DESC) AS rn,"
        "sum(length(content)+128) OVER (ORDER BY id DESC) AS running_chars "
        "FROM messages WHERE session_id=$1 AND id>$2 "
        "AND role IN ('user','assistant')) "
        "SELECT id,role,content FROM ranked "
        "WHERE running_chars<=$3 OR rn=1 ORDER BY id",
        session_id, after_id, char_budget,
    )


async def upsert_context_compaction(
    session_id: int, summary: str, through_message_id: int
) -> None:
    """Advance compaction cursor monotonically; stale writers cannot rewind it."""
    p = await pool()
    await p.execute(
        "INSERT INTO session_context_compactions "
        "(session_id,summary,through_message_id,prompt_version) VALUES ($1,$2,$3,$4) "
        "ON CONFLICT (session_id) DO UPDATE SET "
        "summary=EXCLUDED.summary,through_message_id=EXCLUDED.through_message_id,"
        "prompt_version=EXCLUDED.prompt_version,updated_at=now() "
        "WHERE session_context_compactions.through_message_id "
        "< EXCLUDED.through_message_id",
        session_id, summary.strip(), through_message_id, config.PROMPT_VERSION,
    )


async def transcript(session_id: int) -> list[asyncpg.Record]:
    p = await pool()
    return await p.fetch(
        "SELECT m.id,m.role,m.content FROM messages m "
        "LEFT JOIN review_items r ON r.id=CASE "
        "WHEN m.tool_calls ? 'review_item_id' "
        "THEN (m.tool_calls->>'review_item_id')::bigint ELSE NULL END "
        "WHERE m.session_id=$1 AND m.included_in_extraction "
        "ORDER BY COALESCE(r.source_message_id,m.id),m.id",
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

async def topic_summary(workspace_id: int, topic_id: int) -> Optional[str]:
    p = await pool()
    return await p.fetchval(
        "SELECT summary FROM topic_summaries WHERE workspace_id=$1 AND topic_id=$2",
        workspace_id, topic_id,
    )


async def upsert_topic_summary(
    workspace_id: int, topic_id: int, summary: str
) -> None:
    p = await pool()
    await p.execute(
        "INSERT INTO topic_summaries "
        "(workspace_id,topic_id,summary,prompt_version,generated_at) "
        "VALUES ($1,$2,$3,$4,now()) "
        "ON CONFLICT (workspace_id,topic_id) DO UPDATE SET summary=EXCLUDED.summary, "
        "prompt_version=EXCLUDED.prompt_version, generated_at=now()",
        workspace_id, topic_id, summary, config.PROMPT_VERSION,
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
                "(session_id,workspace_id,topic_id,type,origin,payload,quote,source_message_id, "
                " embedding, duplicate_of, prompt_version, embed_version, "
                " support_mode, grounding_status, grounding_version, "
                " grounding_details) "
                "SELECT $1,s.workspace_id,s.topic_id,$2,$3,$4,$5,$6,$7,$8,$9,$10, "
                "       $11,$12,$13,$14 FROM messages m "
                "JOIN sessions s ON s.id=$1 "
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
    workspace_id: int, topic_id: int, embedding: list[float], current_session: int
) -> Optional[asyncpg.Record]:
    """Closest published fact, plus earlier rows of this extraction run.

    The current extracting session is included to collapse chunk-overlap
    duplicates. Rows from every other unfinished session stay invisible.
    """
    rows = await nearest_canonicals(
        workspace_id, topic_id, embedding, current_session, limit=1
    )
    return rows[0] if rows else None


async def nearest_canonicals(
    workspace_id: int, topic_id: int, embedding: list[float],
    current_session: int, limit: int = 10,
) -> list[asyncpg.Record]:
    """Top-N tenant-local candidates for relation classification."""
    p = await pool()
    return await p.fetch(
        "SELECT e.id, e.payload, e.quote, e.support_mode, "
        "       e.embedding <=> $3 AS dist "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE e.workspace_id=$1 AND e.topic_id=$2 "
        "  AND (s.status='extracted' OR e.session_id=$6) "
        "  AND e.duplicate_of IS NULL AND e.embedding IS NOT NULL "
        "  AND e.embed_version=$4 "
        "  AND e.grounding_version=$5 "
        "  AND e.grounding_status='verified' "
        "  AND (e.embedding <=> $3) <= $7 "
        "ORDER BY dist LIMIT $8",
        workspace_id, topic_id, embedding, config.EMBED_TEXT_VERSION,
        config.GROUNDING_VERSION,
        current_session, config.RAG_MAX_DISTANCE, limit,
    )


async def search_canonical(
    workspace_id: int, topic_id: int, embedding: list[float], limit: int = 5,
    exclude_session: Optional[int] = None,
) -> list[asyncpg.Record]:
    """Top-k canonical facts of the topic nearest a query embedding (JIT context
    for the search_knowledge tool and the auto-RAG STATE block). exclude_session
    drops the current session's own items so auto-RAG shows only OTHER experts."""
    p = await pool()
    return await p.fetch(
        "SELECT e.id,e.topic_id,e.type,e.payload,e.quote,e.origin,e.support_mode, "
        "       e.confirmation_count, "
        "       s.expert_name,s.user_id,e.embedding <=> $3 AS dist "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE e.workspace_id=$1 AND e.topic_id=$2 AND s.status='extracted' "
        "  AND e.duplicate_of IS NULL AND e.embedding IS NOT NULL "
        "  AND e.embed_version=$6 "
        "  AND e.grounding_version=$7 "
        "  AND e.grounding_status='verified' "
        "  AND ($5::bigint IS NULL OR e.session_id <> $5) "
        "  AND (e.embedding <=> $3) <= $8 "
        "ORDER BY dist LIMIT $4",
        workspace_id, topic_id, embedding, limit, exclude_session,
        config.EMBED_TEXT_VERSION, config.GROUNDING_VERSION,
        config.RAG_MAX_DISTANCE,
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


async def canonical_for_topic(
    workspace_id: int, topic_id: int
) -> list[asyncpg.Record]:
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
        "             AND ev.workspace_id=$1 AND ev.topic_id=$2 "
        "             AND ev.grounding_version=$4 "
        "             AND p.kind IN ('user_support','confirmation') "
        "           ORDER BY ev.id, p.ord "
        "           LIMIT 4 "
        "         ) x "
        "       ), '[]'::jsonb) AS supports "
        "FROM extracted_items e JOIN sessions s ON s.id = e.session_id "
        "WHERE e.workspace_id=$1 AND e.topic_id=$2 AND s.status='extracted' "
        "  AND e.duplicate_of IS NULL "
        "  AND e.embed_version=$3 "
        "  AND e.grounding_version=$4 "
        "  AND e.grounding_status='verified' "
        "ORDER BY e.id",
        workspace_id, topic_id, config.EMBED_TEXT_VERSION,
        config.GROUNDING_VERSION,
    )


async def all_for_topic(workspace_id: int, topic_id: int) -> list[asyncpg.Record]:
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
        "WHERE e.workspace_id=$1 AND e.topic_id=$2 ORDER BY e.id",
        workspace_id, topic_id,
    )


# --- durable post-processing ------------------------------------------------

async def recover_stale_jobs() -> int:
    p = await pool()
    result = await p.execute(
        "UPDATE postprocess_jobs SET status='retry_wait',worker_id=NULL,"
        "lease_expires_at=NULL,available_at=now(),updated_at=now(),"
        "last_error=COALESCE(last_error,'worker lease expired') "
        "WHERE status='running' AND lease_expires_at < now()"
    )
    return int(result.rsplit(" ", 1)[-1])


async def claim_postprocess_job(
    worker_id: str, lease_seconds: int
) -> Optional[asyncpg.Record]:
    p = await pool()
    return await p.fetchrow(
        "WITH candidate AS (SELECT id FROM postprocess_jobs "
        "WHERE status IN ('queued','retry_wait') AND available_at<=now() "
        "ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1) "
        "UPDATE postprocess_jobs j SET status='running',attempts=attempts+1,"
        "worker_id=$1,started_at=COALESCE(started_at,now()),"
        "lease_expires_at=now()+make_interval(secs=>$2),updated_at=now() "
        "FROM candidate c WHERE j.id=c.id RETURNING j.*",
        worker_id, lease_seconds,
    )


async def heartbeat_job(job_id: int, worker_id: str, lease_seconds: int) -> bool:
    p = await pool()
    row = await p.fetchrow(
        "UPDATE postprocess_jobs SET lease_expires_at=now()+make_interval(secs=>$3),"
        "updated_at=now() WHERE id=$1 AND worker_id=$2 AND status='running' RETURNING id",
        job_id, worker_id, lease_seconds,
    )
    return row is not None


async def succeed_job(job_id: int, worker_id: str) -> bool:
    p = await pool()
    row = await p.fetchrow(
        "UPDATE postprocess_jobs SET status='succeeded',finished_at=now(),"
        "lease_expires_at=NULL,worker_id=NULL,updated_at=now() "
        "WHERE id=$1 AND worker_id=$2 AND status='running' RETURNING id",
        job_id, worker_id,
    )
    return row is not None


async def fail_job(job_id: int, worker_id: str, error: str) -> str:
    p = await pool()
    return await p.fetchval(
        "UPDATE postprocess_jobs SET "
        "status=CASE WHEN attempts>=max_attempts THEN 'dead' ELSE 'retry_wait' END,"
        "available_at=CASE WHEN attempts>=max_attempts THEN available_at ELSE "
        "now()+make_interval(secs=>LEAST(3600,15*power(2,attempts-1)::int)) END,"
        "last_error=$3,lease_expires_at=NULL,worker_id=NULL,"
        "finished_at=CASE WHEN attempts>=max_attempts THEN now() ELSE NULL END,"
        "updated_at=now() WHERE id=$1 AND worker_id=$2 AND status='running' "
        "RETURNING status", job_id, worker_id, error[:4000],
    )


async def release_job(job_id: int, worker_id: str) -> None:
    p = await pool()
    await p.execute(
        "UPDATE postprocess_jobs SET status='retry_wait',available_at=now(),"
        "lease_expires_at=NULL,worker_id=NULL,updated_at=now() "
        "WHERE id=$1 AND worker_id=$2 AND status='running'", job_id, worker_id,
    )


@asynccontextmanager
async def topic_advisory_lock(topic_id: int):
    """Cross-process lock covering dedup and summary rebuild for one topic."""
    p = await pool()
    async with p.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock($1::bigint)", topic_id)
        try:
            yield
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1::bigint)", topic_id)


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
