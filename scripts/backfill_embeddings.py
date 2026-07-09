"""Backfill embeddings for extracted_items collected while embeddings were off.

Existing verified rows can have embedding IS NULL, so search_knowledge can't
find them. This recomputes each from the same doc text the insert path uses
(retrieval_question + payload + verified expert support spans) and writes it
back. Legacy/partial/needs_review rows stay unpublished and are skipped.

Run: python -m scripts.backfill_embeddings           # only rows with NULL vector
     python -m scripts.backfill_embeddings --stale   # rows whose embed_version
                                                     # != config.EMBED_TEXT_VERSION
                                                     # (the right call after a
                                                     # _embed_text format change)
     python -m scripts.backfill_embeddings --all     # re-embed EVERY row (brute
                                                     # force; --stale is cheaper)

After changing _embed_text's format, bump config.EMBED_TEXT_VERSION and run
--stale: old and new vectors live in different spaces, so a mixed base breaks
dedup until every row is re-embedded. The rewritten rows are stamped with the
current version, so a half-finished run is safe to resume.
"""
import asyncio
import json
import sys

from src import config, db, embed
from src.jobs.extract import _embed_text


async def main() -> None:
    if not embed.enabled():
        print("embeddings disabled (set EMBED_MODE=bundled or external) - nothing to do")
        return
    p = await db.pool()
    if "--all" in sys.argv:
        where, args = (
            "WHERE e.grounding_status='verified' AND e.grounding_version=$1",
            (config.GROUNDING_VERSION,),
        )
    elif "--stale" in sys.argv:
        where, args = (
            "WHERE e.grounding_status='verified' AND e.grounding_version=$1 "
            "AND e.embed_version IS DISTINCT FROM $2",
            (config.GROUNDING_VERSION, config.EMBED_TEXT_VERSION),
        )
    else:
        where, args = (
            "WHERE e.grounding_status='verified' AND e.grounding_version=$1 "
            "AND e.embedding IS NULL",
            (config.GROUNDING_VERSION,),
        )
    rows = await p.fetch(
        "SELECT e.id, e.payload, "
        "       COALESCE(( "
        "         SELECT jsonb_agg(substring(m.content FROM pr.start_char + 1 "
        "                                    FOR pr.end_char - pr.start_char) "
        "                          ORDER BY pr.ord) "
        "         FROM extracted_item_provenance pr "
        "         JOIN messages m ON m.id=pr.message_id "
        "         WHERE pr.item_id=e.id "
        "           AND pr.kind IN ('user_support','confirmation') "
        "       ), '[]'::jsonb) AS support_quotes "
        f"FROM extracted_items e {where}",
        *args,
    )
    print(f"{len(rows)} rows to backfill (target version {config.EMBED_TEXT_VERSION})")
    done = 0
    for r in rows:
        payload = r["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        support_quotes = r["support_quotes"]
        if isinstance(support_quotes, str):
            support_quotes = json.loads(support_quotes)
        vec = await embed.embed(
            _embed_text(payload, support_quotes), required=True
        )
        await p.execute(
            "UPDATE extracted_items SET embedding=$2, embed_version=$3 WHERE id=$1",
            r["id"], vec, config.EMBED_TEXT_VERSION,
        )
        done += 1
    print(f"backfilled {done}/{len(rows)}")
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
