# OmniVoice voices (in-browser)

Pick a voice from the ElevenLabs shared library (names, filters), type text, and OmniVoice speaks it in that voice.
**All speech models run in the browser** (WebGPU + ONNX Runtime Web); they load when the page loads.

Flow when you press Speak: fetch the selected voice's sample → **transcribe it on the fly** (Whisper, ONNX,
transformers.js) → encode it with the OmniVoice codec encoder → clone it for your text → play.

## Layout
- `public/index.html`, `public/js/` — the app (`omnivoice.js` = JS port of upstream's decoding loop, `asr.js`, `audio.js`).
- `app.py` — serves `public/` and proxies two ElevenLabs calls (`/api/voices`, `/api/preview/{id}`) so the API key
  stays server-side. Nothing heavy runs on the server.
- `public/vad.html` — Silero VAD page, hidden/not linked (future speech-to-text input).
- `omnivoice_tts/` + `get_voices.py tts` — the Python reference implementation (ONNX Runtime CPU). It is what was used to
  validate the approach; not used by the web app.

## Run / deploy
```bash
npm ci --ignore-scripts && node scripts/vendor.mjs     # copies ORT-Web / VAD assets to public/vendor
pip install -r requirements.txt
ELEVENLABS_API_KEY=... uvicorn app:app --port 10000
```
Render: `render.yaml` (Docker, free plan, set `ELEVENLABS_API_KEY` in the dashboard).

## Models (downloaded by the browser, cached via the Cache API)
- OmniVoice: `ct03/omnivoice-onnx-int8hq` (LM ~640 MB, codec encoder ~395 MB, decoder ~85 MB).
- ASR: `onnx-community/whisper-base`.
- First visit downloads >1 GB.

## Why not `onnx-community/OmniVoice-Onnx`
Its `llm_decoder.onnx` is causal (tested: changing the last input position leaves all earlier outputs unchanged), while
upstream runs the LM with a full attention mask. With it, Whisper could not recover the input text from the audio. With
the ct03 export (explicit 4D attention-mask input) Whisper recovered the exact text in every test.

## What is verified vs not
Verified (Python reference, CPU):
- ct03 backbone output is intelligible (Whisper round-trip, 3/3 prompts; cloned-voice prompt 1/1).
- On-the-fly transcript of a 9.8 s sample was correct; codec encoder needs input length a multiple of 960 samples.
- JS tokenizer ids and duration estimator are identical to the Python ones on test strings.

**NOT verified:**
- The browser code has never been run (no GPU/browser run of the models in the dev sandbox): WebGPU execution, speed,
  memory, the transformers.js Whisper `dtype` settings, language auto-detect, and the Docker/Render deploy.
- That a clone sounds like the reference speaker (only intelligibility was checked).
- ElevenLabs response field names other than voice_id/name/description/preview_url, and filter values (gender, age...).
- Python reference speed on CPU was ~75 s for 5 s of cloned audio.

## Licensing / use
- ct03's export is licensed `other` (bundles the Higgs Audio 2 community licence); k2-fsa's OmniVoice card says CC-BY-NC
  and prohibits unauthorised voice cloning/impersonation. Check these before hosting publicly, and ElevenLabs' terms
  before cloning library voices.
