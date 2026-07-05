"""Download the local embedding model (harrier-oss-v1-270m) to ./models.

Run: python -m scripts.download_model
Downloads weights only — serving is separate (text-embeddings-inference).
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

REPO = "microsoft/harrier-oss-v1-270m"
DEST = "models/harrier-oss-v1-270m"


def main() -> None:
    path = snapshot_download(REPO, local_dir=DEST)
    print(f"downloaded {REPO} -> {path}")


if __name__ == "__main__":
    main()
