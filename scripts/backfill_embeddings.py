"""Backfill embeddings for extracted_items collected while embeddings were off.

Existing rows have embedding IS NULL, so search_knowledge can't find them. This
recomputes each from the same doc text the insert path uses (retrieval_question
+ payload + quote) and writes it back. Idempotent: only touches NULL rows.

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
        print("embeddings disabled (set EMBED_BASE_URL) — nothing to do")
        return
    p = await db.pool()
    if "--all" in sys.argv:
        where, args = "", ()
    elif "--stale" in sys.argv:
        where, args = "WHERE embed_version IS DISTINCT FROM $1", (config.EMBED_TEXT_VERSION,)
    else:
        where, args = "WHERE embedding IS NULL", ()
    rows = await p.fetch(
        f"SELECT id, payload, quote FROM extracted_items {where}", *args
    )
    print(f"{len(rows)} rows to backfill (target version {config.EMBED_TEXT_VERSION})")
    done = 0
    for r in rows:
        payload = r["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        vec = await embed.embed(_embed_text(payload, r["quote"] or ""))
        if vec is None:
            print(f"  skip #{r['id']}: embed failed")
            continue
        await p.execute(
            "UPDATE extracted_items SET embedding=$2, embed_version=$3 WHERE id=$1",
            r["id"], vec, config.EMBED_TEXT_VERSION,
        )
        done += 1
    print(f"backfilled {done}/{len(rows)}")
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
