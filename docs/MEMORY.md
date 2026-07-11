# Долговечная память

Источник истины: `messages` → grounded `extracted_items` → `memory_claims` с
`memory_claim_evidence`. Граф не является отдельным datastore. PostgreSQL хранит
только проверяемые связи `duplicate_of`, `supports`, `contradicts`, `refines`,
`depends_on`.

Retrieval:

1. vector top-20;
2. PostgreSQL FTS top-20;
3. Reciprocal Rank Fusion;
4. фильтр workspace/current version/status;
5. максимум один verified relation hop;
6. 5–8 claims с исходной цитатой и distinct contributors.

`PUBLIC_COLLECTION_SLUG=public` направляет новых Telegram-пользователей в общий
workspace. Legacy/CLI-сессии остаются приватными. Membership проверяется и в БД,
и перед retrieval. `ADMIN_TELEGRAM_USER_IDS` задаёт администраторов.

Безопасный rollout:

```powershell
python -m src.db migrate
python -m scripts.backfill_memory
python -m scripts.audit_memory
python -m scripts.eval_retrieval tests/fixtures/retrieval_golden.jsonl
python -m scripts.report_retrieval_shadow
```

Сначала держать `HYBRID_RAG_SHADOW=true`, `HYBRID_RAG_ENABLED=false`. Заполнить
golden JSONL реальными `workspace_id`, `topic_id`, `expected_item_ids` и
`expected_relation_types`. Включать hybrid только после выигрыша по recall,
citation correctness и приемлемой latency.

Entity alias index строится для новых jobs при `ENTITY_INDEX_ENABLED=true`.
Исторический backfill с LLM запускается явно: `python -m scripts.backfill_memory
--entities`.

Иерархические summaries opt-in: `HIERARCHICAL_SUMMARIES_ENABLED=true`. Они
строятся как дерево над claims, но не заменяют claims/evidence. Полный entity KG,
Neo4j и многошаговый GraphRAG не используются до подтверждённой потребности.
