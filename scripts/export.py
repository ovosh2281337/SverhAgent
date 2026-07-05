"""Export the knowledge base of a topic to a single Markdown file — the
business-facing artifact (hand to a methodologist, feed to a RAG pipeline,
attach to a report). Canonical items only, grouped by type, with reliability
signals (confirmations, origin, contradictions) and quote provenance.

Run: python -m scripts.export [topic] [outfile]
Default: topic 'default' -> kb_default.md
"""
import asyncio
import json
import sys
from datetime import date

from src import db


def _payload(r) -> dict:
    p = r["payload"]
    return json.loads(p) if isinstance(p, str) else p


def _reliability(r) -> str:
    tags = []
    if r["origin"] == "confirmed_hypothesis":
        tags.append("подтверждённая гипотеза — слабый вес")
    if r["confirmation_count"] > 1:
        tags.append(f"подтверждено ×{r['confirmation_count']}")
    return f" _({'; '.join(tags)})_" if tags else ""


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
            out.append(f"  - источник: {r['expert_name']} — «{r['quote'][:220]}»")
        out.append("")

    if qas:
        out += ["## Вопрос-ответ", ""]
        for r in qas:
            p = _payload(r)
            out.append(f"- **{p.get('question','')}**{_reliability(r)}")
            out.append(f"  - {p.get('answer','')}")
            out.append(f"  - источник: {r['expert_name']}")
        out.append("")

    if terms:
        out += ["## Термины", ""]
        for r in terms:
            p = _payload(r)
            out.append(
                f"- **{p.get('term','')}** — {p.get('definition','')}"
                f"{_reliability(r)} _(источник: {r['expert_name']})_"
            )
        out.append("")

    return "\n".join(out)


async def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "default"
    outfile = sys.argv[2] if len(sys.argv) > 2 else f"kb_{topic}.md"
    rows = await db.canonical_for_topic(topic)
    summary = await db.topic_summary(topic)
    if not rows:
        print(f"по теме '{topic}' записей нет")
        await db.close()
        return
    text = _render(topic, rows, summary)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"экспортировано {len(rows)} записей -> {outfile}")
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
