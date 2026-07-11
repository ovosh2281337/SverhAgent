"""Tenant-safe flat and hybrid retrieval with relation-aware one-hop recall."""
import json
import time
from typing import Any

from . import config, db, embed


def route_query(query: str) -> str:
    """Cheap deterministic router; no extra model call on every retrieval."""
    text = query.casefold()
    if any(word in text for word in ("противореч", "конфликт", "disagree")):
        return "contradictions"
    if any(word in text for word in ("изменил", "изменилось", "раньше", "history")):
        return "temporal"
    if any(word in text for word in ("кто сказал", "кто говорил", "источник", "source")):
        return "provenance"
    if any(word in text for word in ("главные темы", "общая картина", "main themes")):
        return "aggregate"
    return "fact"


def _payload(value: Any) -> dict:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _basic_row(row) -> dict:
    data = dict(row)
    data["payload"] = _payload(data["payload"])
    data["claim_id"] = None
    data["relation_type"] = None
    data["user_ids"] = [data.get("user_id")]
    return data


async def _basic(
    workspace_id: int, topic_id: int, vector: list[float] | None,
    *, exclude_session: int | None, limit: int,
) -> list[dict]:
    if vector is None:
        return []
    rows = await db.search_canonical(
        workspace_id, topic_id, vector, limit=limit,
        exclude_session=exclude_session,
    )
    return [_basic_row(row) for row in rows]


async def _hybrid(
    workspace_id: int, topic_id: int, query: str,
    vector: list[float] | None, *, limit: int, route: str = "fact",
) -> list[dict]:
    p = await db.pool()
    vector_sql = (
        "SELECT id,row_number() OVER (ORDER BY embedding <=> $3) AS rank "
        "FROM memory_claims WHERE workspace_id=$1 AND status='verified' "
        "AND valid_to IS NULL AND grounding_version=$4 AND embed_version=$5 "
        "AND embedding IS NOT NULL AND (embedding <=> $3) <= $6 "
        "ORDER BY embedding <=> $3 LIMIT $7"
        if vector is not None else
        "SELECT NULL::bigint id,NULL::bigint rank "
        "WHERE $3::vector IS NOT NULL AND $5::text IS NOT NULL "
        "AND $6::float8 IS NOT NULL AND $7::int IS NOT NULL AND false"
    )
    sql = f"""
      WITH vector_hits AS ({vector_sql}),
      text_hits AS (
        SELECT id,row_number() OVER (
          ORDER BY ts_rank_cd(search_vector,websearch_to_tsquery('simple',$2)) DESC
        ) AS rank
        FROM memory_claims
        WHERE workspace_id=$1 AND status='verified' AND valid_to IS NULL
          AND grounding_version=$4
          AND search_vector @@ websearch_to_tsquery('simple',$2)
        ORDER BY ts_rank_cd(search_vector,websearch_to_tsquery('simple',$2)) DESC
        LIMIT $8
      ), fused AS (
        SELECT id,sum(score) score FROM (
          SELECT id,1.0/($9+rank) score FROM vector_hits
          UNION ALL
          SELECT id,1.0/($9+rank) score FROM text_hits
        ) hits GROUP BY id
      )
      SELECT c.id AS claim_id,c.topic_id,c.type,c.origin,c.payload,
             c.normalized_statement,c.canonical_extracted_item_id AS id,
             canonical.quote,canonical.support_mode,
             fused.score + CASE WHEN c.topic_id=$10 THEN 0.005 ELSE 0 END AS score,
             evidence.user_ids,evidence.expert_names,
             cardinality(evidence.user_ids) AS confirmation_count,
             NULL::text AS relation_type
      FROM fused JOIN memory_claims c ON c.id=fused.id
      JOIN extracted_items canonical ON canonical.id=c.canonical_extracted_item_id
      JOIN LATERAL (
        SELECT array_agg(DISTINCT ce.user_id) AS user_ids,
               string_agg(DISTINCT s.expert_name, ', ') AS expert_names
        FROM memory_claim_evidence ce
        JOIN sessions s ON s.id=ce.session_id
        WHERE ce.claim_id=c.id
      ) evidence ON true
      ORDER BY score DESC,c.id LIMIT $11
    """
    rows = await p.fetch(
        sql, workspace_id, query, vector, config.GROUNDING_VERSION,
        config.EMBED_TEXT_VERSION, config.RAG_MAX_DISTANCE,
        config.HYBRID_VECTOR_LIMIT, config.HYBRID_FTS_LIMIT,
        config.HYBRID_RRF_K, topic_id, max(limit, 1),
    )
    seeds = [dict(row) for row in rows]
    for row in seeds:
        row["payload"] = _payload(row["payload"])
        row["expert_name"] = row.pop("expert_names") or "unknown"
        row["user_id"] = row["user_ids"][0] if row["user_ids"] else None
        row["retrieval_route"] = route
    if not seeds:
        return []
    related = await p.fetch(
        "WITH seed(id,ord) AS (SELECT * FROM unnest($2::bigint[]) WITH ORDINALITY), "
        "links AS ("
        " SELECT r.target_claim_id id,r.relation_type,seed.ord "
        " FROM seed JOIN memory_claim_relations r ON r.source_claim_id=seed.id "
        " WHERE r.workspace_id=$1 AND r.status='verified' "
        "   AND r.relation_type=ANY($5::text[]) "
        " UNION ALL "
        " SELECT r.source_claim_id id,r.relation_type,seed.ord "
        " FROM seed JOIN memory_claim_relations r ON r.target_claim_id=seed.id "
        " WHERE r.workspace_id=$1 AND r.status='verified' "
        "   AND r.relation_type=ANY($5::text[])) "
        "SELECT DISTINCT ON (c.id) c.id AS claim_id,c.topic_id,c.type,c.origin,"
        "c.payload,c.normalized_statement,c.canonical_extracted_item_id AS id,"
        "e.quote,e.support_mode,CASE WHEN c.status='superseded' "
        "THEN 'superseded' ELSE links.relation_type END AS relation_type,"
        "array_agg(DISTINCT ce.user_id) AS user_ids,"
        "string_agg(DISTINCT s.expert_name, ', ') AS expert_names,"
        "count(DISTINCT ce.user_id)::int AS confirmation_count "
        "FROM links JOIN memory_claims c ON c.id=links.id "
        "JOIN extracted_items e ON e.id=c.canonical_extracted_item_id "
        "JOIN memory_claim_evidence ce ON ce.claim_id=c.id "
        "JOIN sessions s ON s.id=ce.session_id "
        "WHERE (($4::bool AND c.status IN ('verified','superseded')) "
        "OR (c.status='verified' AND c.valid_to IS NULL)) "
        "GROUP BY c.id,e.id,links.relation_type,links.ord "
        "ORDER BY c.id,links.ord,CASE links.relation_type "
        " WHEN 'contradicts' THEN 0 WHEN 'refines' THEN 1 ELSE 2 END "
        "LIMIT $3",
        workspace_id, [row["claim_id"] for row in seeds], limit,
        route == "temporal",
        (["contradicts"] if route == "contradictions" else
         ["contradicts", "supports", "refines", "depends_on"]),
    )
    seen = {row["claim_id"] for row in seeds}
    linked: list[dict] = []
    for record in related:
        row = dict(record)
        if row["claim_id"] in seen:
            for seed in seeds:
                if seed["claim_id"] == row["claim_id"]:
                    seed["relation_type"] = row["relation_type"]
                    break
            continue
        row["payload"] = _payload(row["payload"])
        row["expert_name"] = row.pop("expert_names") or "unknown"
        row["user_id"] = row["user_ids"][0] if row["user_ids"] else None
        row["score"] = 0.0
        row["retrieval_route"] = route
        linked.append(row)
        seen.add(row["claim_id"])
    if not linked or limit <= 1:
        return seeds[:limit]
    # Graph recall must not disappear merely because flat retrieval filled the
    # result limit. Keep the strongest seed for query relevance, then reserve
    # slots for verified one-hop relations, dropping lowest-ranked flat seeds.
    linked_count = min(len(linked), limit - 1)
    seed_count = limit - linked_count
    return seeds[:seed_count] + linked[:linked_count]


async def _log_shadow(
    workspace_id: int, topic_id: int, session_id: int | None, query: str,
    basic: list[dict], hybrid: list[dict], basic_ms: float, hybrid_ms: float,
    error: str | None,
) -> None:
    p = await db.pool()
    await p.execute(
        "INSERT INTO retrieval_shadow_logs "
        "(workspace_id,topic_id,session_id,query,basic_item_ids,hybrid_claim_ids,"
        " basic_latency_ms,hybrid_latency_ms,error) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
        workspace_id, topic_id, session_id, query[:4000],
        [row["id"] for row in basic],
        [row["claim_id"] for row in hybrid if row.get("claim_id")],
        basic_ms, hybrid_ms, error,
    )


async def retrieve_context(
    workspace_id: int, topic_id: int, user_id: int, query: str, *,
    session_id: int | None = None, limit: int | None = None,
) -> list[dict]:
    """Unified recall API; flat vector path always remains a safe fallback."""
    p = await db.pool()
    allowed = await p.fetchval(
        "SELECT EXISTS(SELECT 1 FROM workspace_members "
        "WHERE workspace_id=$1 AND user_id=$2)",
        workspace_id, user_id,
    )
    if not allowed:
        raise PermissionError("user is not a member of the requested workspace")
    limit = limit or config.HYBRID_RESULT_LIMIT
    route = route_query(query)
    vector = None
    if embed.enabled() and query.strip():
        vector = await embed.embed(query, query=True)
    t0 = time.perf_counter()
    basic = await _basic(
        workspace_id, topic_id, vector,
        exclude_session=session_id, limit=limit,
    )
    for row in basic:
        row["retrieval_route"] = route
    basic_ms = (time.perf_counter() - t0) * 1000
    if not config.HYBRID_RAG_ENABLED and not config.HYBRID_RAG_SHADOW:
        return basic
    t1 = time.perf_counter()
    error = None
    try:
        hybrid = await _hybrid(
            workspace_id, topic_id, query, vector, limit=limit, route=route
        )
    except Exception as exc:
        hybrid = []
        error = f"{type(exc).__name__}: {exc}"[:2000]
    hybrid_ms = (time.perf_counter() - t1) * 1000
    if config.HYBRID_RAG_SHADOW:
        try:
            await _log_shadow(
                workspace_id, topic_id, session_id, query, basic, hybrid,
                basic_ms, hybrid_ms, error,
            )
        except Exception:
            pass
    if config.HYBRID_RAG_ENABLED and hybrid:
        return hybrid
    return basic
