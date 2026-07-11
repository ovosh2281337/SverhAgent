"""Re-ground legacy extracted sessions after migration 004.

Migration 004 intentionally makes old extracted_items invisible to RAG/summary:
their exact quote proves location only, not semantic support. This script runs
the new grounded extraction pipeline over already-extracted legacy sessions and
then rebuilds summaries for touched topics.

Run:
    python -m scripts.reground_legacy --dry-run
    python -m scripts.reground_legacy --topic default --verbose
    python -m scripts.reground_legacy --limit 3 --no-summary
"""
import argparse
import asyncio

from src import config, db
from src.jobs import extract, summary


async def _legacy_sessions(topic: str | None) -> list[dict]:
    p = await db.pool()
    rows = await p.fetch(
        """
        SELECT s.id,s.workspace_id,s.topic_id,s.topic,s.expert_name,
               count(old.id) AS legacy_items
        FROM sessions s
        JOIN extracted_items old ON old.session_id=s.id
        WHERE s.status='extracted'
          AND old.grounding_status='legacy'
          AND ($1::text IS NULL OR s.topic=$1)
          AND NOT EXISTS (
              SELECT 1 FROM extracted_items current
              WHERE current.session_id=s.id
                AND current.grounding_version=$2
                AND current.grounding_status<>'legacy'
          )
        GROUP BY s.id,s.workspace_id,s.topic_id,s.topic,s.expert_name
        ORDER BY s.id
        """,
        topic, config.GROUNDING_VERSION,
    )
    return [dict(row) for row in rows]


async def _run(args) -> int:
    sessions = await _legacy_sessions(args.topic)
    if args.limit is not None:
        sessions = sessions[:args.limit]

    if not sessions:
        print("legacy sessions to re-ground: 0")
        return 0

    print(f"legacy sessions to re-ground: {len(sessions)}")
    for s in sessions:
        print(
            f"  #{s['id']} topic={s['topic']} legacy={s['legacy_items']} "
            f"expert={s['expert_name']}"
        )
    if args.dry_run:
        return 0

    touched_topics: set[tuple[int, int, str]] = set()
    ok = 0
    failed = 0
    for s in sessions:
        trace: list[str] | None = [] if args.verbose else None
        try:
            n = await extract.run(s["id"], trace, reprocess_legacy=True)
            ok += 1
            touched_topics.add((s["workspace_id"], s["topic_id"], s["topic"]))
            print(f"#{s['id']}: verified={n}")
            if trace:
                print("\n".join(trace))
        except Exception as exc:
            failed += 1
            print(f"#{s['id']}: FAILED {type(exc).__name__}: {exc}")

    if not args.no_summary:
        for workspace_id, topic_id, topic in sorted(touched_topics):
            try:
                rebuilt = await summary.run(workspace_id, topic_id)
                print(f"summary[{topic}]: {'rebuilt' if rebuilt else 'skipped'}")
            except Exception as exc:
                failed += 1
                print(f"summary[{topic}]: FAILED {type(exc).__name__}: {exc}")

    print(f"done: sessions_ok={ok}, failures={failed}")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", help="only re-ground one topic")
    ap.add_argument("--limit", type=int, help="process at most N sessions")
    ap.add_argument("--dry-run", action="store_true", help="show sessions only")
    ap.add_argument("--verbose", action="store_true", help="print extraction trace")
    ap.add_argument("--no-summary", action="store_true", help="skip summary rebuild")
    args = ap.parse_args()

    async def _main() -> None:
        try:
            raise SystemExit(await _run(args))
        finally:
            await db.close()

    asyncio.run(_main())


if __name__ == "__main__":
    main()
