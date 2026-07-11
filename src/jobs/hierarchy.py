"""Opt-in hierarchical aggregate memory over verified claims."""
from .. import config, db, llm


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


async def rebuild(workspace_id: int, topic_id: int) -> int:
    """Atomically replace summary tree; abort if claim snapshot changed."""
    p = await db.pool()
    claims = await p.fetch(
        "SELECT id,normalized_statement FROM memory_claims "
        "WHERE workspace_id=$1 AND topic_id=$2 AND status='verified' "
        "AND valid_to IS NULL ORDER BY id",
        workspace_id, topic_id,
    )
    if not claims:
        return 0
    levels: list[list[dict]] = []
    leaves = []
    leaf_size = max(1, config.HIERARCHY_LEAF_SIZE)
    branch_size = max(2, config.HIERARCHY_BRANCH_SIZE)
    for number, batch in enumerate(
        _chunks(list(claims), leaf_size)
    ):
        inputs = [
            {"id": f"claim:{row['id']}", "text": row["normalized_statement"]}
            for row in batch
        ]
        leaves.append({
            "key": f"leaf-{number}",
            "summary": await llm.hierarchical_memory_summary(inputs),
            "claim_ids": [row["id"] for row in batch],
            "children": [],
        })
    levels.append(leaves)
    previous = leaves
    level = 1
    while len(previous) > 1:
        current = []
        for number, batch in enumerate(
            _chunks(previous, branch_size)
        ):
            inputs = [
                {"id": f"summary:{item['key']}", "text": item["summary"]}
                for item in batch
            ]
            current.append({
                "key": f"level-{level}-{number}",
                "summary": await llm.hierarchical_memory_summary(inputs),
                "claim_ids": [],
                "children": [item["key"] for item in batch],
            })
        levels.append(current)
        previous = current
        level += 1

    async with p.acquire() as conn, conn.transaction():
        current_ids = await conn.fetch(
            "SELECT id FROM memory_claims WHERE workspace_id=$1 AND topic_id=$2 "
            "AND status='verified' AND valid_to IS NULL ORDER BY id",
            workspace_id, topic_id,
        )
        if [row["id"] for row in current_ids] != [row["id"] for row in claims]:
            raise RuntimeError("claims changed during hierarchical summarization")
        await conn.execute(
            "DELETE FROM memory_summary_nodes WHERE workspace_id=$1 AND topic_id=$2",
            workspace_id, topic_id,
        )
        ids: dict[str, int] = {}
        for level_number, nodes in enumerate(levels):
            for node in nodes:
                node_id = await conn.fetchval(
                    "INSERT INTO memory_summary_nodes "
                    "(workspace_id,topic_id,level,cluster_key,summary,prompt_version) "
                    "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    workspace_id, topic_id, level_number, node["key"],
                    node["summary"], config.HIERARCHY_PROMPT_VERSION,
                )
                ids[node["key"]] = node_id
                for claim_id in node["claim_ids"]:
                    await conn.execute(
                        "INSERT INTO memory_summary_claims "
                        "(workspace_id,topic_id,summary_id,claim_id) "
                        "VALUES ($1,$2,$3,$4)",
                        workspace_id, topic_id, node_id, claim_id,
                    )
                for child_key in node["children"]:
                    await conn.execute(
                        "INSERT INTO memory_summary_children "
                        "(workspace_id,topic_id,parent_summary_id,child_summary_id) "
                        "VALUES ($1,$2,$3,$4)",
                        workspace_id, topic_id, node_id, ids[child_key],
                    )
    return sum(len(nodes) for nodes in levels)
