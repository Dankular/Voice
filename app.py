"""Web service: static VAD page + OmniVoice voice-design TTS API.

  GET  /healthz         liveness (does not load the model)
  GET  /api/attributes  filter vocabulary (gender, age, pitch, style, accent, dialect)
  POST /api/tts         JSON {text, gender?, age?, pitch?, whisper?, accent?, dialect?, language?, speed?, seed?}
                        -> audio/wav (24 kHz, mono)
"""
import asyncio
import io
import os
import threading
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from omnivoice_tts.attributes import FACETS, build_instruct
from omnivoice_tts.engine import SAMPLE_RATE, OmniVoiceONNX
from omnivoice_tts.models import ensure_models

MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "300"))
THREADS = int(os.environ.get("TTS_THREADS", "0"))

app = FastAPI(title="OmniVoice voice design")
_engine: Optional[OmniVoiceONNX] = None
_lock = threading.Lock()  # one synthesis at a time; the model is CPU-bound


def get_engine() -> OmniVoiceONNX:
    global _engine
    if _engine is None:
        model_dir = os.environ.get("OMNIVOICE_MODEL_DIR")
        higgs_dir = os.environ.get("OMNIVOICE_HIGGS_DIR")
        if not (model_dir and higgs_dir):
            model_dir, higgs_dir = ensure_models()
        _engine = OmniVoiceONNX(model_dir, higgs_dir, num_threads=THREADS)
    return _engine


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    gender: Optional[str] = None
    age: Optional[str] = None
    pitch: Optional[str] = None
    whisper: bool = False
    accent: Optional[str] = None
    dialect: Optional[str] = None
    language: Optional[str] = None
    speed: float = Field(1.0, gt=0.25, lt=4)
    seed: Optional[int] = None


def _synthesize(req: TTSRequest) -> bytes:
    instruct = build_instruct(req.gender, req.age, req.pitch, "whisper" if req.whisper else None,
                              req.accent, req.dialect)
    with _lock:
        wav = get_engine().synthesize(req.text, instruct, req.language, speed=req.speed, seed=req.seed)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/attributes")
def attributes():
    return FACETS


@app.post("/api/tts")
async def tts(req: TTSRequest):
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, f"text too long (max {MAX_CHARS} characters)")
    try:
        data = await asyncio.get_running_loop().run_in_executor(None, _synthesize, req)
    except ValueError as e:  # invalid/conflicting attributes
        raise HTTPException(422, str(e))
    return Response(data, media_type="audio/wav")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "public", html=True), name="static")
