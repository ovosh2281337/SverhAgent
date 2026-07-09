"""OpenAI-compatible /v1/embeddings server for harrier-oss-v1-270m.

Loads the local sentence-transformers model (last-token pooling, L2 norm, dim
640, cosine) and exposes the response shape expected by the OpenAI SDK.

Run:
    python -m scripts.serve_embed

Then use:
    EMBED_MODE=bundled
    EMBED_BASE_URL=http://127.0.0.1:8300/v1
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_DIR = os.getenv("EMBED_MODEL_DIR", "models/harrier-oss-v1-270m")
PORT = int(os.getenv("EMBED_PORT", "8300"))  # 8044-8143/8144-8243 are Windows-reserved


def _resolve_device() -> str:
    requested = os.getenv("EMBED_DEVICE", "auto").strip().lower() or "auto"
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("EMBED_DEVICE must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError("EMBED_DEVICE=cuda but torch is not importable") from exc
        print("CUDA check unavailable, using CPU for bundled embeddings")
        return "cpu"

    has_cuda = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "EMBED_DEVICE=cuda but CUDA is unavailable; "
                "set EMBED_DEVICE=auto or EMBED_DEVICE=cpu"
            )
        return "cuda"

    if has_cuda:
        print("CUDA available, using CUDA for bundled embeddings")
        return "cuda"
    print("CUDA unavailable, using CPU for bundled embeddings; startup and latency may be slower")
    return "cpu"


_device = _resolve_device()
print(f"loading {MODEL_DIR} on {_device} ...")
_model = SentenceTransformer(MODEL_DIR, device=_device)
print(f"loaded. dim={_model.get_sentence_embedding_dimension()}")

app = FastAPI()


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "dim": _model.get_sentence_embedding_dimension(),
        "device": _device,
        "model_dir": MODEL_DIR,
    }


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest) -> dict:
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
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
