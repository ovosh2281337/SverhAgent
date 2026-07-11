"""Durable semantic-memory projection over verified extracted evidence."""
import json
from datetime import datetime, timezone
from typing import Any

from . import config, db, llm


def statement_from_payload(payload: Any) -> str:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return str(payload).strip()
    text = (
        payload.get("statement")
        or payload.get("answer")
        or payload.get("definition")
        or json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    qualifiers = payload.get("qualifiers")
    return f"{text} ({qualifiers})".strip() if qualifiers else str(text).strip()


def normalize_name(value: str) -> str:
    return " ".join(value.casefold().strip().split())[:300]


async def _ensure_claim(conn, item_id: int):
    item = await conn.fetchrow(
        "SELECT e.*,s.user_id FROM extracted_items e "
        "JOIN sessions s ON s.id=e.session_id WHERE e.id=$1",
        item_id,
    )
    if item is None or item["grounding_status"] != "verified":
        return None
    canonical_id = item["duplicate_of"] or item["id"]
    canonical = item if canonical_id == item["id"] else await conn.fetchrow(
        "SELECT e.*,s.user_id FROM extracted_items e "
        "JOIN sessions s ON s.id=e.session_id "
        "WHERE e.id=$1 AND e.workspace_id=$2 AND e.topic_id=$3",
        canonical_id, item["workspace_id"], item["topic_id"],
    )
    if canonical is None or canonical["grounding_status"] != "verified":
        return None
    statement = statement_from_payload(canonical["payload"])
    claim = await conn.fetchrow(
        "INSERT INTO memory_claims "
        "(workspace_id,topic_id,canonical_extracted_item_id,type,origin,"
        " normalized_statement,payload,embedding,embed_version,grounding_version,"
        " prompt_version,observed_at,valid_from) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$12) "
        "ON CONFLICT (workspace_id,topic_id,canonical_extracted_item_id) "
        "DO UPDATE SET normalized_statement=EXCLUDED.normalized_statement,"
        "payload=EXCLUDED.payload,embedding=EXCLUDED.embedding,"
        "updated_at=now() RETURNING *",
        canonical["workspace_id"], canonical["topic_id"], canonical["id"],
        canonical["type"], canonical["origin"], statement, canonical["payload"],
        canonical["embedding"], canonical["embed_version"],
        canonical["grounding_version"], canonical["prompt_version"],
        canonical["created_at"],
    )
    evidence_rows = await conn.fetch(
        "SELECT e.id,e.session_id,s.user_id FROM extracted_items e "
        "JOIN sessions s ON s.id=e.session_id "
        "WHERE e.workspace_id=$1 AND e.topic_id=$2 "
        "  AND (e.id=$3 OR e.duplicate_of=$3) "
        "  AND e.grounding_status='verified' AND e.grounding_version=$4",
        canonical["workspace_id"], canonical["topic_id"], canonical["id"],
        config.GROUNDING_VERSION,
    )
    for evidence in evidence_rows:
        await conn.execute(
            "INSERT INTO memory_claim_evidence "
            "(workspace_id,topic_id,claim_id,extracted_item_id,session_id,user_id) "
            "VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING",
            claim["workspace_id"], claim["topic_id"], claim["id"], evidence["id"],
            evidence["session_id"], evidence["user_id"],
        )
    return claim


async def project_session(
    session_id: int, *, index_entities: bool | None = None
) -> int:
    """Idempotently materialize verified claims, evidence, relations, entities."""
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        items = await conn.fetch(
            "SELECT e.id,e.grounding_details,e.support_mode,e.created_at,s.user_id "
            "FROM extracted_items e JOIN sessions s ON s.id=e.session_id "
            "WHERE e.session_id=$1 AND e.grounding_status='verified' "
            "  AND e.grounding_version=$2 ORDER BY e.id",
            session_id, config.GROUNDING_VERSION,
        )
        projected = []
        for item in items:
            claim = await _ensure_claim(conn, item["id"])
            if claim is not None:
                projected.append((item, claim))

        for item, source_claim in projected:
            details = item["grounding_details"]
            if isinstance(details, str):
                details = json.loads(details)
            decision = (details or {}).get("memory_relation") or {}
            relation = decision.get("relation")
            target_item_id = decision.get("target_item_id")
            if relation not in {
                "supports", "contradicts", "refines", "depends_on"
            } or not isinstance(target_item_id, int):
                continue
            target_claim = await _ensure_claim(conn, target_item_id)
            if target_claim is None or target_claim["id"] == source_claim["id"]:
                continue
            status = "verified" if decision.get("verified") else "needs_review"
            await conn.execute(
                "INSERT INTO memory_claim_relations "
                "(workspace_id,source_claim_id,target_claim_id,relation_type,status,"
                " confidence,classifier_version,verifier_version,details) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT DO NOTHING",
                source_claim["workspace_id"], source_claim["id"], target_claim["id"],
                relation, status, float(decision.get("confidence") or 0),
                config.RELATION_CLASSIFIER_VERSION,
                config.RELATION_VERIFIER_VERSION,
                json.dumps(decision, ensure_ascii=False),
            )
            if status == "verified" and item["support_mode"] == "correction":
                contributors = await conn.fetch(
                    "SELECT DISTINCT user_id FROM memory_claim_evidence "
                    "WHERE claim_id=$1",
                    target_claim["id"],
                )
                if {row["user_id"] for row in contributors} == {item["user_id"]}:
                    await conn.execute(
                        "UPDATE memory_claims SET status='superseded',valid_to=$2,"
                        "superseded_by=$3,updated_at=now() "
                        "WHERE id=$1 AND valid_to IS NULL",
                        target_claim["id"], item["created_at"], source_claim["id"],
                    )

    should_index = (
        config.ENTITY_INDEX_ENABLED if index_entities is None else index_entities
    )
    if should_index and projected:
        await _index_entities(projected)
    return len({claim["id"] for _, claim in projected})


async def backfill_claims(*, index_entities: bool = False) -> dict[str, int]:
    """Idempotently project every current verified session into semantic memory."""
    p = await db.pool()
    session_ids = await p.fetch(
        "SELECT DISTINCT session_id FROM extracted_items "
        "WHERE grounding_status='verified' AND grounding_version=$1 "
        "ORDER BY session_id",
        config.GROUNDING_VERSION,
    )
    claims = 0
    for row in session_ids:
        claims += await project_session(
            row["session_id"], index_entities=index_entities
        )
    return {"sessions": len(session_ids), "projected_claims": claims}


async def _index_entities(projected) -> None:
    statements = [claim["normalized_statement"] for _, claim in projected]
    try:
        entity_lists = await llm.extract_memory_entities(statements)
    except Exception:
        return
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        for (_, claim), names in zip(projected, entity_lists):
            for display in names[:12]:
                normalized = normalize_name(display)
                if len(normalized) < 2:
                    continue
                entity = await conn.fetchrow(
                    "INSERT INTO memory_entities "
                    "(workspace_id,normalized_name,display_name,embed_version) "
                    "VALUES ($1,$2,$3,$4) ON CONFLICT (workspace_id,normalized_name) "
                    "DO UPDATE SET display_name=EXCLUDED.display_name,updated_at=now() "
                    "RETURNING *",
                    claim["workspace_id"], normalized, display[:300],
                    config.EMBED_TEXT_VERSION,
                )
                await conn.execute(
                    "INSERT INTO memory_entity_aliases "
                    "(workspace_id,entity_id,alias,normalized_alias) "
                    "VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING",
                    claim["workspace_id"], entity["id"], display[:300], normalized,
                )
                await conn.execute(
                    "INSERT INTO memory_claim_entities "
                    "(workspace_id,claim_id,entity_id,mention) "
                    "VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING",
                    claim["workspace_id"], claim["id"], entity["id"], display[:300],
                )


async def supersede_claim(
    old_claim_id: int, new_claim_id: int, *, valid_at: datetime | None = None
) -> None:
    """Close old validity interval without deleting historical evidence."""
    when = valid_at or datetime.now(timezone.utc)
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        old = await conn.fetchrow(
            "SELECT workspace_id,topic_id FROM memory_claims WHERE id=$1",
            old_claim_id,
        )
        new = await conn.fetchrow(
            "SELECT workspace_id,topic_id FROM memory_claims WHERE id=$1",
            new_claim_id,
        )
        if old is None or new is None or tuple(old) != tuple(new):
            raise ValueError("claims must exist in the same workspace/topic")
        await conn.execute(
            "UPDATE memory_claims SET status='superseded',valid_to=$2,"
            "superseded_by=$3,updated_at=now() "
            "WHERE id=$1 AND valid_to IS NULL",
            old_claim_id, when, new_claim_id,
        )
