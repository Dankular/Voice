"""Static hosting + a thin ElevenLabs proxy. All speech models run in the browser (see public/js).

  GET /api/voices?gender=&accent=&age=&language=&use_case=&search=&page=   shared-voice library (names etc.)
  GET /api/preview/{voice_id}                                              the voice's sample audio (same-origin)
  GET /healthz

Needs ELEVENLABS_API_KEY on the server; it never reaches the browser.
"""
import asyncio
import contextlib
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from get_voices import fetch_page

MAX_PREVIEW_BYTES = 8 * 1024 * 1024
SKIP_KEYS = {"preview_url", "public_owner_id", "date_unix"}   # not shown; previews are served via /api/preview


def _scalar_tags(v: dict) -> dict:
    """Every simple field of a library entry, plus entries of a nested `labels` dict. The page builds its filters
    from whatever tags are present, so no field list is hard-coded (the live response shape is unverified here)."""
    out = {}
    for k, val in list(v.items()) + list((v.get("labels") or {}).items() if isinstance(v.get("labels"), dict) else []):
        if k in SKIP_KEYS or k.endswith("_url") or isinstance(val, (dict, list)):
            continue
        out.setdefault(k, val)
    return out


ROOT = Path(__file__).parent
LIST_TTL = 600                      # seconds a library listing is served from memory
KEEPALIVE_S = 600                   # Render's free tier spins an idle service down after ~15 min
_list_cache: dict = {}              # (sorted query items) -> (expires_at, payload)


def _warm_files():
    """Read the big static files once so the first visitor's download is not served from cold disk."""
    t, n = time.time(), 0
    for sub in ("public/models", "public/vendor"):
        for f in (ROOT / sub).rglob("*"):
            if f.is_file():
                with open(f, "rb") as fh:
                    while chunk := fh.read(8 << 20):
                        n += len(chunk)
    print(f"[warm] read {n / 1e6:.0f} MB of static files in {time.time() - t:.1f}s", flush=True)


async def _prefetch_voices():
    """Fill the listing cache (and the preview table, which is empty after every restart) for the page's first request."""
    if not os.environ.get("ELEVENLABS_API_KEY"):
        return
    try:
        await _list({"page": 0, "page_size": 100})
        print("[warm] voice list cached", flush=True)
    except Exception as e:             # never block startup on the upstream API
        print(f"[warm] voice list prefetch skipped: {e}", flush=True)


async def _keepalive():
    """Ping our own public URL so the platform sees inbound traffic (set KEEPALIVE=0 to disable)."""
    url = os.environ.get("KEEPALIVE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if not url or os.environ.get("KEEPALIVE", "1") == "0":
        return
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(KEEPALIVE_S)
        try:
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=20).read())
        except Exception as e:
            print(f"[keepalive] ping failed: {e}", flush=True)


@contextlib.asynccontextmanager
async def lifespan(_app):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _warm_files)
    tasks = [asyncio.create_task(_prefetch_voices()), asyncio.create_task(_keepalive())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="OmniVoice voices", lifespan=lifespan)
_previews: dict = {}   # voice_id -> preview_url, filled only from library responses (clients can't pick URLs)


def _key() -> str:
    k = os.environ.get("ELEVENLABS_API_KEY")
    if not k:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set on the server")
    return k


@app.get("/healthz")
def healthz():
    return "ok"


async def _list(params: dict) -> dict:
    key = tuple(sorted(params.items()))
    hit = _list_cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    try:
        data = await asyncio.get_running_loop().run_in_executor(None, fetch_page, _key(), params)
    except SystemExit as e:           # fetch_page reports HTTP errors this way
        raise HTTPException(502, str(e))
    out = []
    for v in data.get("voices", []):
        if v.get("voice_id") and v.get("preview_url"):
            _previews[v["voice_id"]] = v["preview_url"]
        out.append(_scalar_tags(v))
    payload = {"voices": out, "has_more": data.get("has_more", False)}
    _list_cache[key] = (time.time() + LIST_TTL, payload)
    return payload


@app.get("/api/voices")
async def voices(gender: Optional[str] = None, accent: Optional[str] = None, age: Optional[str] = None,
                 language: Optional[str] = None, use_case: Optional[str] = None, search: Optional[str] = None,
                 page: int = 0, page_size: int = Query(100, le=100)):
    params = {"gender": gender, "accent": accent, "age": age, "language": language, "use_cases": use_case,
              "search": search, "page": page, "page_size": page_size}
    return await _list({k: v for k, v in params.items() if v not in (None, "")})


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


class Static(StaticFiles):
    """Version-pinned vendor files and model parts never change: let browsers cache them for a year."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if path.startswith(("vendor/", "models/kv/")):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp


app.mount("/", Static(directory=Path(__file__).parent / "public", html=True), name="static")
