"""Compare flat vector recall with hybrid claim recall on JSONL golden cases."""
import argparse
import asyncio
import json
import time
from pathlib import Path

from src import config, db, embed, retrieval


def _recall(expected: set[int], rows: list[dict]) -> float:
    if not expected:
        return 1.0
    found = {row["id"] for row in rows}
    return len(expected & found) / len(expected)


async def _main(path: Path) -> None:
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    totals = {
        "cases": len(cases), "flat_recall": 0.0, "hybrid_recall": 0.0,
        "contradiction_recall": 0.0, "citation_correctness": 0.0,
        "flat_latency_ms": 0.0, "hybrid_latency_ms": 0.0,
    }
    for case in cases:
        vector = (
            await embed.embed(case["query"], query=True)
            if embed.enabled() else None
        )
        started = time.perf_counter()
        flat = await retrieval._basic(
            case["workspace_id"], case["topic_id"], vector,
            exclude_session=None, limit=case.get("limit", 8),
        )
        totals["flat_latency_ms"] += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        hybrid = await retrieval._hybrid(
            case["workspace_id"], case["topic_id"], case["query"], vector,
            limit=case.get("limit", 8),
        )
        totals["hybrid_latency_ms"] += (time.perf_counter() - started) * 1000
        expected = set(case.get("expected_item_ids", []))
        totals["flat_recall"] += _recall(expected, flat)
        totals["hybrid_recall"] += _recall(expected, hybrid)
        expected_relations = set(case.get("expected_relation_types", []))
        found_relations = {
            row.get("relation_type") for row in hybrid if row.get("relation_type")
        }
        totals["contradiction_recall"] += (
            len(expected_relations & found_relations) / len(expected_relations)
            if expected_relations else 1.0
        )
        totals["citation_correctness"] += float(
            all(row.get("quote") and row.get("id") for row in hybrid)
        )
    divisor = max(1, len(cases))
    for key in totals:
        if key != "cases":
            totals[key] = round(totals[key] / divisor, 4)
    totals["grounding_version"] = config.GROUNDING_VERSION
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    p = await db.pool()
    await p.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("golden", type=Path)
    asyncio.run(_main(parser.parse_args().golden))
