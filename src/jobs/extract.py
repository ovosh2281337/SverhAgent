"""Extraction job: transcript -> facts / qa_pairs / terms, each quote-anchored.

Idempotency is structural, not a count check: the job runs only if it wins the
atomic CAS finished->extracting (db.claim_for_extraction). A double finish can
never trigger a double extraction. Re-extraction with a new prompt is an
explicit operation (wipe the session's derived rows, rerun with a new version).

On insert each item is deduped against the topic's canonical facts via pgvector:
  dist <= DEDUP_SAME  -> same fact: store as duplicate + bump confirmation_count
  DEDUP_SAME..NEAR    -> candidate contradiction: LLM check, tag if conflicting
  dist  > DEDUP_NEAR  -> new canonical fact
Without embeddings (no EMBED_BASE_URL) every item is inserted as canonical.
"""
import asyncio
import json

from .. import config, db, embed, llm

_TYPES = {"fact", "qa_pair", "term"}
_ORIGINS = {"expert_claim", "confirmed_hypothesis"}

# Split a long transcript into chunks before extraction. A dense session makes
# the model emit a huge JSON array and burn thousands of reasoning tokens, then
# hit the server's output cap mid-array (LengthFinishReasonError). Per-chunk
# extraction keeps each call small enough to finish. Trade-off: cross-chunk
# self-contradiction detection weakens, so keep chunks large (usually 1 chunk).
_CHUNK_CHARS = 6000
# Carry the last few messages of a chunk into the start of the next one, so a
# question/answer split across the boundary is still extractable from both sides
# (the extractor sees the full exchange in at least one chunk). Dedup-on-insert
# collapses any item that both chunks emit, so overlap costs a little tokens, not
# duplicate facts.
_CHUNK_OVERLAP = 2


def _line(r) -> str:
    who = "ЭКСПЕРТ" if r["role"] == "user" else "АГЕНТ"
    return f"[#{r['id']}] {who}: {r['content']}"


def _render_transcript(rows) -> tuple[str, set[int]]:
    expert_ids = {r["id"] for r in rows if r["role"] == "user"}
    return "\n".join(_line(r) for r in rows), expert_ids


def _chunks(rows) -> list[str]:
    """Group consecutive rows into rendered chunks under the char budget, with a
    small message overlap between adjacent chunks so an exchange split across a
    boundary stays intact in at least one chunk. A single oversized message
    becomes its own chunk (salvage still guards it)."""
    lines = [_line(r) for r in rows]
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for ln in lines:
        if cur and size + len(ln) > _CHUNK_CHARS:
            out.append("\n".join(cur))
            cur = cur[-_CHUNK_OVERLAP:]  # seed next chunk with the tail overlap
            size = sum(len(x) + 1 for x in cur)
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def _embed_text(payload: dict, quote: str) -> str:
    # Plain natural text, no JSON syntax: light embedding models (harrier is a
    # 270m encoder) are trained on prose, and {"key": ...} braces are semantic
    # noise. The retrieval question goes first so question-form queries land
    # close (doc2query). NB: changing this format shifts every vector — re-embed
    # the whole base afterwards (python -m scripts.backfill_embeddings --all),
    # otherwise dedup compares old-format vs new-format vectors.
    body = (
        payload.get("statement")
        or f"{payload.get('question', '')} {payload.get('answer', '')}".strip()
        or f"{payload.get('term', '')} — {payload.get('definition', '')}".strip("— ")
    )
    parts = [payload.get("retrieval_question", ""), body]
    q = payload.get("qualifiers")
    if q:
        parts.append(f"Условия: {q}")
    parts.append(quote)
    return ". ".join(p.strip() for p in parts if p and p.strip())


async def _insert_with_dedup(
    session_id: int, topic: str, type_: str, origin: str,
    payload: dict, quote: str, src: int | None,
) -> None:
    vec = await embed.embed(_embed_text(payload, quote))
    dup_of = None
    if vec is not None:
        near = await db.nearest_canonical(topic, vec)
        if near is not None:
            dist = float(near["dist"])
            if dist <= config.DEDUP_SAME:
                await db.bump_confirmation(near["id"])
                dup_of = near["id"]  # store as duplicate for provenance
            elif dist <= config.DEDUP_NEAR:
                near_payload = near["payload"]
                if isinstance(near_payload, str):
                    near_payload = json.loads(near_payload)
                conflict = await llm.contradiction(
                    _embed_text(near_payload, near["quote"]),
                    _embed_text(payload, quote),
                )
                if conflict:
                    payload = {**payload, "contradicts": near["id"]}
    await db.add_extracted_item(
        session_id, type_, origin, payload, quote, src,
        embedding=vec, duplicate_of=dup_of,
    )


async def run(session_id: int) -> int:
    if not await db.claim_for_extraction(session_id):
        return 0  # not finished, or already extracting/extracted
    try:
        return await _run_claimed(session_id)
    except Exception:
        # A crash mid-extraction (LLM/network/DB) must not strand the session
        # in 'extracting' forever. Wipe the partial inserts and put the status
        # back so a retry starts from a clean slate.
        await db.wipe_extracted(session_id)
        await db.revert_extraction(session_id)
        raise


async def _run_claimed(session_id: int) -> int:
    sess = await db.get_session(session_id)
    topic = sess["topic"] if sess else "default"
    rows = await db.transcript(session_id)
    if not rows:
        await db.mark_extracted(session_id)
        return 0
    _, expert_ids = _render_transcript(rows)

    items: list[dict] = []
    for chunk in _chunks(rows):
        items.extend(await llm.extract(chunk))
    inserted = 0
    for it in items:
        type_ = it.get("type")
        quote = (it.get("quote") or "").strip()
        payload = it.get("payload")
        if type_ not in _TYPES or not quote or not isinstance(payload, dict):
            continue  # schema violation -> drop
        origin = it.get("origin")
        origin = origin if origin in _ORIGINS else "expert_claim"
        if it.get("contradicts_self"):
            payload = {**payload, "contradicts_self": True}
        rq = it.get("retrieval_question")
        if isinstance(rq, str) and rq.strip():
            payload = {**payload, "retrieval_question": rq.strip()}
        ref = it.get("source_ref")
        src = ref if isinstance(ref, int) and ref in expert_ids else None
        await _insert_with_dedup(
            session_id, topic, type_, origin, payload, quote, src
        )
        inserted += 1

    await db.mark_extracted(session_id)
    return inserted


if __name__ == "__main__":
    import sys

    async def _main():
        sid = int(sys.argv[1])
        n = await run(sid)
        print(f"extracted {n} items from session {sid}")
        await db.close()

    asyncio.run(_main())
