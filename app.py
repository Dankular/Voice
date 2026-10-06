"""Static hosting + a thin ElevenLabs proxy. All speech models run in the browser (see public/js).

  GET /api/voices?gender=&accent=&age=&language=&use_case=&search=&page=   shared-voice library (names etc.)
  GET /api/preview/{voice_id}                                              the voice's sample audio (same-origin)
  GET /healthz

Needs ELEVENLABS_API_KEY on the server; it never reaches the browser.
"""
import asyncio
import os
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from get_voices import fetch_page

MAX_PREVIEW_BYTES = 8 * 1024 * 1024
# fields beyond the first four are not verified against a live response (no API key in the dev sandbox)
LIST_KEYS = ["voice_id", "name", "description", "preview_url", "gender", "accent", "age", "descriptive",
             "use_case", "language", "category"]

app = FastAPI(title="OmniVoice voices")
_previews: dict = {}   # voice_id -> preview_url, filled only from library responses (clients can't pick URLs)


def _key() -> str:
    k = os.environ.get("ELEVENLABS_API_KEY")
    if not k:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set on the server")
    return k


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/voices")
async def voices(gender: Optional[str] = None, accent: Optional[str] = None, age: Optional[str] = None,
                 language: Optional[str] = None, use_case: Optional[str] = None, search: Optional[str] = None,
                 page: int = 0, page_size: int = Query(24, le=100)):
    params = {"gender": gender, "accent": accent, "age": age, "language": language, "use_cases": use_case,
              "search": search, "page": page, "page_size": page_size}
    try:
        data = await asyncio.get_running_loop().run_in_executor(
            None, fetch_page, _key(), {k: v for k, v in params.items() if v not in (None, "")})
    except SystemExit as e:           # fetch_page reports HTTP errors this way
        raise HTTPException(502, str(e))
    out = []
    for v in data.get("voices", []):
        if v.get("voice_id") and v.get("preview_url"):
            _previews[v["voice_id"]] = v["preview_url"]
        out.append({k: v.get(k) for k in LIST_KEYS})
    return {"voices": out, "has_more": data.get("has_more", False)}


def _fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("preview URL must be https")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read(MAX_PREVIEW_BYTES + 1)
    if len(data) > MAX_PREVIEW_BYTES:
        raise ValueError("preview audio too large")
    return data


@app.get("/api/preview/{voice_id}")
async def preview(voice_id: str):
    url = _previews.get(voice_id)
    if not url:
        raise HTTPException(404, "unknown voice_id - list voices first")
    try:
        data = await asyncio.get_running_loop().run_in_executor(None, _fetch, url)
    except Exception as e:
        raise HTTPException(502, f"could not fetch preview: {e}")
    return Response(data, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "public", html=True), name="static")
