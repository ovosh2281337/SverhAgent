"""Export the knowledge base of a topic to a single Markdown file — the
business-facing artifact (hand to a methodologist, feed to a RAG pipeline,
attach to a report). Canonical items only, grouped by type, with reliability
signals (confirmations, origin, contradictions) and quote provenance.

Run: python -m scripts.export WORKSPACE_ID TOPIC_ID [outfile]
"""
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from src import db


def _payload(r) -> dict:
    p = r["payload"]
    return json.loads(p) if isinstance(p, str) else p


def _json(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        return json.loads(value)
    return value


def _reliability(r) -> str:
    tags = []
    if r["origin"] == "confirmed_hypothesis":
        tags.append("подтверждённая гипотеза — слабый вес")
    if r["support_mode"]:
        tags.append(f"support={r['support_mode']}")
    if r["confirmation_count"] > 1:
        tags.append(f"подтверждено ×{r['confirmation_count']}")
    return f" _({'; '.join(tags)})_" if tags else ""


def _support_lines(r, limit: int = 2) -> list[str]:
    supports = _json(r["supports"], [])
    lines: list[str] = []
    for s in supports[:limit]:
        quote = (s.get("quote") or "").strip()
        if not quote:
            continue
        if len(quote) > 220:
            quote = quote[:220] + "…"
        who = s.get("expert_name") or r["expert_name"]
        kind = s.get("kind") or "support"
        lines.append(f"  - evidence/{kind}: {who} — «{quote}»")
    if not lines and r["quote"]:
        quote = r["quote"][:220] + ("…" if len(r["quote"]) > 220 else "")
        lines.append(f"  - evidence: {r['expert_name']} — «{quote}»")
    return lines


def _render(topic: str, rows, summary: str | None) -> str:
    facts = [r for r in rows if r["type"] == "fact"]
    qas = [r for r in rows if r["type"] == "qa_pair"]
    terms = [r for r in rows if r["type"] == "term"]

    out = [
        f"# База знаний: {topic}",
        f"_Экспорт {date.today().isoformat()} · {len(rows)} канонических записей "
        f"({len(facts)} фактов, {len(qas)} Q&A, {len(terms)} терминов)_",
        "",
    ]
    if summary:
        out += ["## Сводка", "", summary, ""]

    if facts:
        out += ["## Факты", ""]
        for r in facts:
            p = _payload(r)
            line = p.get("statement", "")
            q = p.get("qualifiers")
            if q:
                line += f" **[{q}]**"
            out.append(f"- {line}{_reliability(r)}")
            if p.get("contradicts") or p.get("contradicts_self"):
                out.append("  - ⚠ есть противоречие с другой записью")
            out.extend(_support_lines(r))
        out.append("")

    if qas:
        out += ["## Вопрос-ответ", ""]
        for r in qas:
            p = _payload(r)
            out.append(f"- **{p.get('question','')}**{_reliability(r)}")
            out.append(f"  - {p.get('answer','')}")
            out.extend(_support_lines(r))
        out.append("")

    if terms:
        out += ["## Термины", ""]
        for r in terms:
            p = _payload(r)
            out.append(
                f"- **{p.get('term','')}** — {p.get('definition','')}"
                f"{_reliability(r)} _(источник: {r['expert_name']})_"
            )
            out.extend(_support_lines(r))
        out.append("")

    return "\n".join(out)


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m scripts.export WORKSPACE_ID TOPIC_ID [outfile]")
    workspace_id, topic_id = int(sys.argv[1]), int(sys.argv[2])
    topic_row = await db.get_topic(workspace_id, topic_id)
    if topic_row is None:
        raise SystemExit("topic does not belong to workspace")
    topic = topic_row["name"]
    outfile = sys.argv[3] if len(sys.argv) > 3 else f"kb_{workspace_id}_{topic_id}.md"
    rows = await db.canonical_for_topic(workspace_id, topic_id)
    summary = await db.topic_summary(workspace_id, topic_id)
    if not rows:
        print(f"по теме '{topic}' записей нет")
        await db.close()
        return
    text = _render(topic, rows, summary)
    outpath = Path(outfile)
    if outpath.parent != Path("."):
        outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"экспортировано {len(rows)} записей -> {outfile}")
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
