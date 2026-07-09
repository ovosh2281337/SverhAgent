"""Download/cache optional voice STT models used by scripts.serve_stt.

Run:
    python -m scripts.download_stt_model
"""

from __future__ import annotations

import os
import shutil


def _resolve_device() -> str:
    requested = os.getenv("STT_DEVICE", "auto").strip().lower() or "auto"
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("STT_DEVICE must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError("STT_DEVICE=cuda but torch is not importable") from exc
        return "cpu"

    has_cuda = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "STT_DEVICE=cuda but CUDA is unavailable; "
                "set STT_DEVICE=auto or STT_DEVICE=cpu"
            )
        return "cuda"
    return "cuda" if has_cuda else "cpu"


def main() -> None:
    import gigaam
    from silero_vad import load_silero_vad

    model_name = os.getenv("STT_MODEL", "v3_e2e_rnnt")
    device = _resolve_device()
    print(f"loading GigaAM {model_name} on {device} ...")
    gigaam.load_model(model_name, device=device)
    print("loading Silero VAD ...")
    load_silero_vad(onnx=True)
    if shutil.which("ffmpeg") is None:
        print("warning: ffmpeg not found in PATH; scripts.serve_stt will need it")
    print("STT models cached")


if __name__ == "__main__":
    main()
