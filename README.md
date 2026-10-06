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

## Speed
Per step the LM runs twice (conditional + unconditional guidance). What the code does about cost:
- **GPU-resident post-processing:** LM logits stay on the GPU and go through a tiny ONNX graph (`public/models/post_*.onnx`,
  built by `scripts/make_postproc.py`) that does the guidance mix, log-softmax, argmax and confidence; only ~8×T values
  are read back (previously 2 × ~13 MB logits plus JS exp/log loops per step).
- **Shape bucketing:** sequence lengths are padded to multiples of 64/32 so WebGPU kernels compiled for one utterance can
  be reused. Padding is masked out of attention; verified identical to unpadded (max logit diff 0.0, Python/CPU).
- **Reused tensors:** attention mask / position ids are built once per utterance instead of every forward.
- **Sentence chunking + queued playback:** audio starts after the first sentence; a live readout shows RTF
  (<1 = faster than real time) and per-step LM / post / JS ms, with step 1 shown separately (kernel-compile cost).
- **Knobs:** steps (default 16), guidance on/off (off ≈ 2× faster, quality effect not measured), voice-sample length.
- Not possible: KV caching (bidirectional attention means every position changes each step).

Measured (Python/CPU, one prompt, intelligibility only via Whisper round-trip): 32, 16 and 8 steps all exact; a 5 s voice
sample dropped the first words, 9.8 s was fine (default sample cap is 12 s). Naturalness/similarity at fewer steps is
**not** assessed. No WebGPU timings exist yet — use the on-page readout.

## Filters
Built from the tags on the returned voices (gender, age, accent, style, use case, ... whatever is present): chips with live
counts, OR within a tag, AND across tags, plus name/description search. Tag names are not hard-coded because the live
response shape is unverified here.

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
