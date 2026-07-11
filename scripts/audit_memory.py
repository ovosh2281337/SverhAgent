"""Fail-fast isolation, provenance, and projection audit."""
import asyncio
import json

from src import db


CHECKS = {
    "cross_tenant_duplicates": """
      SELECT count(*) FROM extracted_items e
      JOIN extracted_items d ON d.id=e.duplicate_of
      WHERE (e.workspace_id,e.topic_id)<>(d.workspace_id,d.topic_id)
    """,
    "cross_workspace_relations": """
      SELECT count(*) FROM memory_claim_relations r
      JOIN memory_claims s ON s.id=r.source_claim_id
      JOIN memory_claims t ON t.id=r.target_claim_id
      WHERE r.workspace_id<>s.workspace_id OR r.workspace_id<>t.workspace_id
    """,
    "claims_without_evidence": """
      SELECT count(*) FROM memory_claims c
      WHERE NOT EXISTS (SELECT 1 FROM memory_claim_evidence e WHERE e.claim_id=c.id)
    """,
    "evidence_tenant_mismatch": """
      SELECT count(*) FROM memory_claim_evidence e
      JOIN memory_claims c ON c.id=e.claim_id
      JOIN extracted_items x ON x.id=e.extracted_item_id
      WHERE (e.workspace_id,e.topic_id)<>(c.workspace_id,c.topic_id)
         OR (e.workspace_id,e.topic_id)<>(x.workspace_id,x.topic_id)
    """,
    "telegram_sessions_outside_public": """
      SELECT count(*) FROM sessions s JOIN users u ON u.id=s.user_id
      JOIN workspaces w ON w.id=s.workspace_id
      WHERE u.telegram_user_id IS NOT NULL AND w.slug<>'public'
    """,
    "finalized_without_job": """
      SELECT count(*) FROM sessions s
      WHERE s.status='finalized'
        AND NOT EXISTS (
          SELECT 1 FROM postprocess_jobs j WHERE j.session_id=s.id
        )
    """,
    "expired_running_jobs": """
      SELECT count(*) FROM postprocess_jobs
      WHERE status='running' AND lease_expires_at < now()
    """,
}


async def _main() -> None:
    p = await db.pool()
    result = {name: await p.fetchval(sql) for name, sql in CHECKS.items()}
    result["claims"] = await p.fetchval("SELECT count(*) FROM memory_claims")
    result["evidence"] = await p.fetchval(
        "SELECT count(*) FROM memory_claim_evidence"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    await p.close()
    if any(result[name] for name in CHECKS):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_main())
