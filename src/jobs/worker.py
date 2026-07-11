"""Lease-based durable post-processing worker."""
import asyncio
import logging
import os
import socket
import uuid
from collections.abc import Awaitable, Callable

from .. import config, db, memory
from . import eval as evaljob
from . import extract, hierarchy, summary

log = logging.getLogger("postprocess-worker")
Notify = Callable[[object, list[object]], Awaitable[None]]


async def _heartbeat(job_id: int, worker_id: str, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=config.JOB_HEARTBEAT_SECONDS
            )
        except asyncio.TimeoutError:
            if not await db.heartbeat_job(
                job_id, worker_id, config.JOB_LEASE_SECONDS
            ):
                raise RuntimeError(f"lost lease for postprocess job {job_id}")


async def process(job, notify: Notify | None = None) -> None:
    """Idempotent extraction + summary under one cross-process topic lock."""
    async with db.topic_advisory_lock(job["topic_id"]):
        await extract.run(job["session_id"])
        await memory.project_session(job["session_id"])
        await summary.run(job["workspace_id"], job["topic_id"])
        if config.HIERARCHICAL_SUMMARIES_ENABLED:
            await hierarchy.rebuild(job["workspace_id"], job["topic_id"])
    try:
        await evaljob.run(job["session_id"])
    except Exception:
        log.exception("non-critical eval failed: session=%s", job["session_id"])
    if notify is not None:
        rows = await db.extracted_for_session(job["session_id"])
        await notify(job, rows)


async def run(
    stop: asyncio.Event,
    notify: Notify | None = None,
    *,
    worker_id: str | None = None,
) -> None:
    worker_id = worker_id or (
        f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    )
    recovered = await db.recover_stale_jobs()
    if recovered:
        log.warning("recovered stale postprocess jobs: %s", recovered)
    current = None
    while not stop.is_set():
        current = await db.claim_postprocess_job(worker_id, config.JOB_LEASE_SECONDS)
        if current is None:
            await db.recover_stale_jobs()
            try:
                await asyncio.wait_for(stop.wait(), timeout=config.JOB_POLL_SECONDS)
            except asyncio.TimeoutError:
                pass
            continue
        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            _heartbeat(current["id"], worker_id, heartbeat_stop),
            name=f"job-heartbeat-{current['id']}",
        )
        work = asyncio.create_task(
            process(current, notify), name=f"postprocess-job-{current['id']}"
        )
        try:
            done, _ = await asyncio.wait(
                {work, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                await heartbeat
                raise RuntimeError(f"heartbeat stopped for job {current['id']}")
            await work
            if not await db.succeed_job(current["id"], worker_id):
                raise RuntimeError(f"lost lease before completing job {current['id']}")
        except asyncio.CancelledError:
            await db.release_job(current["id"], worker_id)
            raise
        except Exception as exc:
            status = await db.fail_job(current["id"], worker_id, repr(exc))
            log.exception("postprocess job failed: id=%s status=%s", current["id"], status)
        finally:
            heartbeat_stop.set()
            if not work.done():
                work.cancel()
            await asyncio.gather(work, heartbeat, return_exceptions=True)
            current = None
