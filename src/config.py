import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"missing env var: {name}")
    return val


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
DATABASE_URL = _require("DATABASE_URL")

# --- Chat LLM: OpenAI-compatible endpoint (local server / gateway) -----------
# Point OPENAI_BASE_URL at any OpenAI-compatible /v1 server (vLLM, llama.cpp,
# LM Studio, TGI, ...). No temperature, no max_tokens are ever sent — the server
# defaults decide. API key is often a dummy for local servers.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local")

# NOTE: harrier-oss-v1-270m is an EMBEDDING model — it cannot drive the
# interview. These must point at a chat/instruct model served on the endpoint.
DIALOG_MODEL = os.getenv("DIALOG_MODEL", "local-chat")
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "local-chat")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "local-chat")
EVAL_MODEL = os.getenv("EVAL_MODEL", "local-chat")

# Session token budget (prompt+completion, accumulated in sessions.tokens_used).
# Soft cap injects a "wrap up" nudge into STATE; hard cap forces end. 0 disables.
SOFT_CAP_TOKENS = _int("SOFT_CAP_TOKENS", 120_000)
HARD_CAP_TOKENS = _int("HARD_CAP_TOKENS", 200_000)

# --- Embeddings: local harrier-oss-v1-270m via OpenAI-compatible endpoint -----
# Powers dedup-on-insert + the search_knowledge tool. Optional: leave
# EMBED_BASE_URL empty to run without (items stay canonical, search_knowledge
# returns nothing, collection loop still works). Serve harrier with
# text-embeddings-inference to expose /v1/embeddings. dim = 640 (Gemma3-270m
# hidden size); must match vector(640) in the migration.
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "local")
EMBED_MODEL = os.getenv("EMBED_MODEL", "microsoft/harrier-oss-v1-270m")
EMBED_DIM = 640
# harrier is instruction-tuned: retrieval queries get an instruct prefix, stored
# documents do not (from config_sentence_transformers.json).
EMBED_QUERY_PROMPT = os.getenv(
    "EMBED_QUERY_PROMPT",
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: ",
)
# Version of the _embed_text() formatting (extract.py). Stored per row so we know
# which rows predate a format change and need re-embedding. Bump whenever the
# text fed to the embedder changes shape, then:
#   python -m scripts.backfill_embeddings --stale   # re-embed only outdated rows
# v1 = JSON-syntax doc text; v2 = plain-prose doc text (question + body + quote).
EMBED_TEXT_VERSION = os.getenv("EMBED_TEXT_VERSION", "v2")
# Cosine-distance thresholds for dedup (0 = identical, 2 = opposite):
#   dist <= DEDUP_SAME     -> same fact: mark duplicate + bump confirmation_count
#   DEDUP_SAME..DEDUP_NEAR -> candidate contradiction: LLM check
#   dist >  DEDUP_NEAR     -> new canonical fact
DEDUP_SAME = float(os.getenv("DEDUP_SAME", "0.15"))
DEDUP_NEAR = float(os.getenv("DEDUP_NEAR", "0.35"))

# --- Web search: Tavily (LLM-oriented search) --------------------------------
# Lets the interviewer verify hardware/terms mid-reasoning before building a
# question on its own (possibly wrong) prior. Optional: empty key hides the
# web_search / web_fetch tools. Free tier ~1000 req/mo at app.tavily.com.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Bump when the dialogue system prompt changes, so sessions/extractions record
# which prompt produced them (A/B against the fixed test-set).
PROMPT_VERSION = "v5.1"
