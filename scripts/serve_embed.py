"""Minimal OpenAI-compatible /v1/embeddings server for harrier-oss-v1-270m.

Loads the local sentence-transformers model (last-token pooling + L2 norm, dim
640, cosine) and exposes exactly the shape the openai SDK's embeddings.create
expects. Prompts are NOT applied here: src/embed.py prepends the retrieval
instruct prefix for queries and sends documents raw, so this server encodes
input verbatim (default_prompt_name is null in the model config — no auto-prompt).

Run:  python -m scripts.serve_embed
Then set in .env:  EMBED_BASE_URL=http://localhost:8300/v1
"""
import os
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.getenv("EMBED_MODEL_DIR", "models/harrier-oss-v1-270m")
PORT = int(os.getenv("EMBED_PORT", "8300"))  # 8044-8143/8144-8243 are Windows-reserved

_device = "cuda" if os.getenv("EMBED_DEVICE", "cuda") == "cuda" else "cpu"
print(f"loading {MODEL_DIR} on {_device} …")
_model = SentenceTransformer(MODEL_DIR, device=_device)
print(f"loaded. dim={_model.get_sentence_embedding_dimension()}")

app = FastAPI()


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "dim": _model.get_sentence_embedding_dimension()}


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest) -> dict:
    texts = [req.input] if isinstance(req.input, str) else req.input
    # normalize_embeddings: the model's own Normalize module already L2-norms,
    # but pass it explicitly so cosine == dot regardless of module wiring.
    vecs = _model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    data = [
        {"object": "embedding", "index": i, "embedding": v.tolist()}
        for i, v in enumerate(vecs)
    ]
    n = sum(len(t.split()) for t in texts)
    return {
        "object": "list",
        "data": data,
        "model": req.model or MODEL_DIR,
        "usage": {"prompt_tokens": n, "total_tokens": n},
        "created": int(time.time()),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
