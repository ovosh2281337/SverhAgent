"""Local speech-to-text service for Telegram voice messages (GigaAM v3 + VAD).

Holds the GigaAM v3 e2e-RNNT model (Russian ASR, punctuation + normalization
built in) and Silero VAD in one process; the bot talks to it over HTTP so it
never loads torch/weights itself and inference never blocks the aiogram loop -
same split as scripts/serve_embed.py.

Why VAD-first chunking: GigaAM's .transcribe is only reliable up to 25 s of audio
(longer -> hallucinated loops). Voice messages run minutes, so we split. But
shorter chunks = less context = worse text, so we cut ONLY on real speech pauses
and greedily merge segments up to STT_CHUNK_SEC (< 25 s). The model therefore
never sees silence/noise, and each piece stays inside its reliable window.

Run:  python -m scripts.serve_stt
Then set in .env:  STT_BASE_URL=http://localhost:8301
"""
import os
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

import gigaam
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16000  # GigaAM + Silero both operate at 16 kHz mono
PORT = int(os.getenv("STT_PORT", "8301"))  # 8044-8243 are Windows-reserved
MODEL_NAME = os.getenv("STT_MODEL", "v3_e2e_rnnt")
CHUNK_SEC = float(os.getenv("STT_CHUNK_SEC", "22"))   # < 25 s reliable window
MAX_SEC = float(os.getenv("STT_MAX_SEC", "900"))      # refuse audio longer than this
_MERGE_GAP_SEC = 1.0   # bridge pauses up to this when merging, for context
_MIN_CHUNK_SEC = 0.3   # drop slivers this short - nothing to transcribe


def _resolve_device() -> str:
    requested = os.getenv("STT_DEVICE", "auto").strip().lower() or "auto"
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("STT_DEVICE must be one of: auto, cpu, cuda")
    if requested == "cpu":
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

if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg not found in PATH - required to decode voice audio")

DEVICE = _resolve_device()
print(f"loading GigaAM {MODEL_NAME} on {DEVICE} ...")
_model = gigaam.load_model(MODEL_NAME, device=DEVICE)
print("loading Silero VAD ...")
_vad = load_silero_vad(onnx=True)
print("ready.")

# One model, one GPU: FastAPI runs sync endpoints in a threadpool, so two voice
# messages arriving together would otherwise race on the model / VRAM.
_infer_lock = threading.Lock()

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


def _decode(data: bytes) -> np.ndarray:
    """Any container (Telegram .oga/opus, mp3, wav...) -> 16 kHz mono float32.

    ffmpeg reads the raw upload from stdin and writes headerless f32le to stdout;
    that sidesteps needing torchaudio codecs for the opus Telegram sends."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(proc.stderr.decode("utf-8", "replace")[:300] or "decode failed")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _chunks(segments: list[dict]) -> list[tuple[int, int]]:
    """Greedily merge VAD speech segments into <= CHUNK_SEC windows.

    Segments arrive sorted with sample offsets. Merge a segment into the current
    window when the gap is a short pause AND the total stays inside the reliable
    window; otherwise start a new window. Longer-than-window single segments are
    already split by Silero's max_speech_duration_s, so each input segment fits."""
    cap = int(CHUNK_SEC * SAMPLE_RATE)
    gap = int(_MERGE_GAP_SEC * SAMPLE_RATE)
    out: list[tuple[int, int]] = []
    for seg in segments:
        s, e = int(seg["start"]), int(seg["end"])
        if out and (s - out[-1][1]) <= gap and (e - out[-1][0]) <= cap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _write_wav(samples: np.ndarray) -> str:
    """PCM16 mono wav to a temp path (transcribe() takes a file path only)."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return path


def _transcribe(wav: np.ndarray) -> tuple[str, int]:
    tensor = torch.from_numpy(wav)
    segments = get_speech_timestamps(
        tensor, _vad,
        sampling_rate=SAMPLE_RATE,
        max_speech_duration_s=CHUNK_SEC,
        min_silence_duration_ms=300,
        speech_pad_ms=150,
    )
    windows = _chunks(segments)
    min_len = int(_MIN_CHUNK_SEC * SAMPLE_RATE)
    texts: list[str] = []
    for s, e in windows:
        if e - s < min_len:
            continue
        path = _write_wav(wav[s:e])
        try:
            res = _model.transcribe(path)
        finally:
            os.unlink(path)
        text = (res.text or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts), len(windows)


@app.post("/v1/transcribe")
async def transcribe(request: Request) -> Response:
    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty body"}, status_code=400)
    try:
        wav = _decode(data)
    except Exception as exc:  # noqa: BLE001 - bad upload, report as 400
        return JSONResponse({"error": f"decode: {exc}"}, status_code=400)
    duration = len(wav) / SAMPLE_RATE
    if duration > MAX_SEC:
        return JSONResponse(
            {"error": f"audio too long: {duration:.0f}s > {MAX_SEC:.0f}s"},
            status_code=413,
        )
    with _infer_lock:
        text, n_chunks = _transcribe(wav)
    return JSONResponse({"text": text, "duration_sec": duration, "chunks": n_chunks})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
