"""Operational comparison of flat and hybrid shadow traffic."""
import asyncio
import json

from src import db


async def _main() -> None:
    p = await db.pool()
    row = await p.fetchrow(
        """
        SELECT count(*)::int AS requests,
          count(*) FILTER (WHERE error IS NOT NULL)::int AS errors,
          round(avg(basic_latency_ms)::numeric,2)::float8 AS basic_avg_ms,
          round(avg(hybrid_latency_ms)::numeric,2)::float8 AS hybrid_avg_ms,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY basic_latency_ms)
            AS basic_p95_ms,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY hybrid_latency_ms)
            AS hybrid_p95_ms,
          avg(CASE WHEN cardinality(basic_item_ids || hybrid_claim_ids)=0 THEN 1
              ELSE (SELECT count(*) FROM (
                    SELECT unnest(basic_item_ids) INTERSECT
                    SELECT unnest(hybrid_claim_ids)) overlap)::float
                   / GREATEST(cardinality(basic_item_ids),1) END) AS overlap_ratio
        FROM retrieval_shadow_logs
        """
    )
    result = dict(row)
    for key, value in result.items():
        if isinstance(value, float):
            result[key] = round(value, 4)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    await p.close()


if __name__ == "__main__":
    asyncio.run(_main())
