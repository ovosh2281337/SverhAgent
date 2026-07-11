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


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_set(name: str) -> frozenset[int]:
    return frozenset(
        int(value.strip())
        for value in os.getenv(name, "").split(",")
        if value.strip()
    )


TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
DATABASE_URL = _require("DATABASE_URL")

# Optional zero-onboarding collection. When set, every private Telegram user is
# enrolled into this workspace; empty keeps isolated personal workspaces.
PUBLIC_COLLECTION_SLUG = os.getenv("PUBLIC_COLLECTION_SLUG", "public").strip()
PUBLIC_COLLECTION_NAME = (
    os.getenv("PUBLIC_COLLECTION_NAME", "Public knowledge collection").strip()
    or "Public knowledge collection"
)
ADMIN_TELEGRAM_USER_IDS = _int_set("ADMIN_TELEGRAM_USER_IDS")

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
# Adversarial semantic-grounding judge. It may share an endpoint/model during
# development, but has a separate role so production can isolate it from the
# extractor that proposed the claim.
GROUND_MODEL = os.getenv("GROUND_MODEL", EVAL_MODEL)

# Physical context window. Dialogue length itself is unlimited: old turns are
# compacted automatically when the next prompt approaches this window.
DIALOG_CONTEXT_TOKENS = _int("DIALOG_CONTEXT_TOKENS", 32_768)
if DIALOG_CONTEXT_TOKENS < 4_096:
    raise RuntimeError("DIALOG_CONTEXT_TOKENS must be at least 4096")

# --- Embeddings: harrier-oss-v1-270m via OpenAI-compatible endpoint -----------
# EMBED_MODE separates "which endpoint should the app use" from "should the
# local launcher start a bundled endpoint":
#   disabled -> no vector features, even if EMBED_BASE_URL is accidentally set
#   bundled  -> local scripts/run.ps1 starts scripts.serve_embed
#   external -> app uses EMBED_BASE_URL, launcher does not start anything
_RAW_EMBED_MODE = os.getenv("EMBED_MODE", "").strip().lower()
_RAW_EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "").strip()
if not _RAW_EMBED_MODE:
    EMBED_MODE = "external" if _RAW_EMBED_BASE_URL else "disabled"
elif _RAW_EMBED_MODE in {"disabled", "bundled", "external"}:
    EMBED_MODE = _RAW_EMBED_MODE
else:
    raise RuntimeError("EMBED_MODE must be one of: disabled, bundled, external")

EMBED_PORT = _int("EMBED_PORT", 8300)
if EMBED_MODE == "disabled":
    EMBED_BASE_URL = ""
elif EMBED_MODE == "bundled":
    EMBED_BASE_URL = _RAW_EMBED_BASE_URL or f"http://127.0.0.1:{EMBED_PORT}/v1"
else:
    EMBED_BASE_URL = _RAW_EMBED_BASE_URL
    if not EMBED_BASE_URL:
        raise RuntimeError("EMBED_MODE=external requires EMBED_BASE_URL")
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
# v1 = JSON-syntax doc text; v2 = plain prose + one legacy quote;
# v3 = normalized claim + one verified expert-support span (never agent context).
EMBED_TEXT_VERSION = os.getenv("EMBED_TEXT_VERSION", "v3")
# Semantic provenance contract. Only items verified by this exact contract are
# visible to dedup, RAG and summaries. Bump after changing acceptance rules.
GROUNDING_VERSION = os.getenv("GROUNDING_VERSION", "g1")
# Cosine-distance thresholds for dedup (0 = identical, 2 = opposite):
#   dist <= DEDUP_SAME     -> same fact: mark duplicate + bump confirmation_count
#   DEDUP_SAME..DEDUP_NEAR -> candidate contradiction: LLM check
#   dist >  DEDUP_NEAR     -> new canonical fact
DEDUP_SAME = float(os.getenv("DEDUP_SAME", "0.15"))
DEDUP_NEAR = float(os.getenv("DEDUP_NEAR", "0.35"))
# Retrieval is intentionally looser than dedup: it should find related context,
# but weak nearest neighbours (for example dist~0.7) must not enter the prompt
# as if they were facts relevant to the current turn.
RAG_MAX_DISTANCE = _float("RAG_MAX_DISTANCE", 0.55)

# Layered-memory retrieval. Shadow mode runs both paths but keeps flat vector
# RAG user-visible until evaluation proves the hybrid path.
HYBRID_RAG_ENABLED = _bool("HYBRID_RAG_ENABLED", False)
HYBRID_RAG_SHADOW = _bool("HYBRID_RAG_SHADOW", True)
HYBRID_VECTOR_LIMIT = _int("HYBRID_VECTOR_LIMIT", 20)
HYBRID_FTS_LIMIT = _int("HYBRID_FTS_LIMIT", 20)
HYBRID_RESULT_LIMIT = _int("HYBRID_RESULT_LIMIT", 8)
HYBRID_RRF_K = _int("HYBRID_RRF_K", 60)
RELATION_CLASSIFIER_VERSION = os.getenv("RELATION_CLASSIFIER_VERSION", "r1")
RELATION_VERIFIER_VERSION = os.getenv("RELATION_VERIFIER_VERSION", "rv1")
ENTITY_INDEX_ENABLED = _bool("ENTITY_INDEX_ENABLED", True)
HIERARCHICAL_SUMMARIES_ENABLED = _bool("HIERARCHICAL_SUMMARIES_ENABLED", False)
HIERARCHY_LEAF_SIZE = _int("HIERARCHY_LEAF_SIZE", 24)
HIERARCHY_BRANCH_SIZE = _int("HIERARCHY_BRANCH_SIZE", 8)
HIERARCHY_PROMPT_VERSION = os.getenv("HIERARCHY_PROMPT_VERSION", "hierarchy-v1")

# Durable post-processing worker.
JOB_LEASE_SECONDS = _int("JOB_LEASE_SECONDS", 180)
JOB_HEARTBEAT_SECONDS = _int("JOB_HEARTBEAT_SECONDS", 30)
JOB_POLL_SECONDS = _float("JOB_POLL_SECONDS", 2.0)
if JOB_HEARTBEAT_SECONDS >= JOB_LEASE_SECONDS:
    raise RuntimeError("JOB_HEARTBEAT_SECONDS must be less than JOB_LEASE_SECONDS")

# --- Web search: Tavily (LLM-oriented search) --------------------------------
# Lets the interviewer verify hardware/terms mid-reasoning before building a
# question on its own (possibly wrong) prior. Optional: empty key hides the
# web_search / web_fetch tools. Free tier ~1000 req/mo at app.tavily.com.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- STT: local GigaAM v3 service (scripts/serve_stt.py) ----------------------
# Lets experts answer with Telegram voice messages: audio is chunked by Silero
# VAD and transcribed by GigaAM v3 e2e-RNNT. Empty = voice disabled, the bot asks
# for text. STT_MAX_VOICE_SEC caps voice length (also keeps us under Telegram's
# 20 MB download limit and the model's reliable window budget).
STT_BASE_URL = os.getenv("STT_BASE_URL", "")
STT_MAX_VOICE_SEC = _int("STT_MAX_VOICE_SEC", 600)

# Bump when the dialogue system prompt changes, so sessions/extractions record
# which prompt produced them (A/B against the fixed test-set).
PROMPT_VERSION = "v6.0"
