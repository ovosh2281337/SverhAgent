"""Idempotent verified-item to claim/evidence backfill."""
import argparse
import asyncio

from src import db, memory


async def _main(index_entities: bool) -> None:
    await db.migrate()
    print(await memory.backfill_claims(index_entities=index_entities))
    (await db.pool()).terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entities", action="store_true",
        help="also call the configured LLM to build entity aliases",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.entities))
