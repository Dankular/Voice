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

## Speed / real time
Measured in the browser by the user (WebGPU, old engine): chunk of 1.7 s audio took 31.9 s (RTF 18.5); step 1 6.3 s (kernel
compile), later steps ≈1.6 s each, post-processing 35 ms. So the LM forward was ~everything, and ~80% of its tokens were the
voice sample, recomputed every step.

**Cached-prefix engine** (`public/js/omnivoice_kv.js`, built by `export/build_all.sh`):
- The LM is re-exported as ONE graph with explicit K/V I/O. A full pass over [style | text | voice sample] caches its K/V once;
  every later step runs only the target positions (+ the mask-out of stale target K/V columns). ≈ target/total of the FLOPs.
  This is an approximation (the prefix K/V no longer sees the evolving target tokens); see "Verified" for what was tested.
- Weights are 4-bit MatMulNBits (309 MB; WebGPU has tuned kernels for this op) instead of int8 DequantizeLinear→MatMul. The
  audio output head stays fp32. Text/audio embedding lookups moved to JS (tables: 155 MB int8 text, 33 MB audio).
- GPU-resident logits + post-processing graph, bucketed shapes, one-off GPU warm-up at page load, sentence-chunked playback.
- Guidance (CFG) now costs about as much as a cached step (its pass is target-only, no cache), so it defaults to OFF.
- Defaults: 8 steps, guidance off, 12 s voice sample. The page shows RTF and per-step timings (full pass vs cached steps).
- Model files (~480 MB) are committed under `public/models/kv/` as parts ≤ 80 MB (git hosts reject larger files) with a
  `manifest.json` (sizes + sha256); the page downloads the parts in parallel, stitches them, and the browser caches them.
  Served same-origin by `app.py` with long-lived cache headers. `?kv=<base url>` points at any host serving the whole files.
  Regenerate with `export/build_all.sh` (then copy `kv_out/parts/*` into `public/models/kv/`). Note: this adds ~480 MB to git
  history; moving the files to a Hugging Face repo later would not remove them from history.
- If the new model files are not reachable, the app falls back to the old (slow) engine and says so.

**Projection, not a measurement:** from the old per-step timings, a 1.7 s chunk should drop from ~32 s to a few seconds on the
same GPU, i.e. RTF in the low single digits, approaching/under 1 for longer chunks and faster GPUs. Real time on every GPU is
not guaranteed. Measure with the on-page readout.

Other optimisations kept: shape bucketing (padding verified identical), reused tensors, GPU-side post-processing graph
(`scripts/make_postproc.py`, verified vs numpy).

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

## Verified (cached-prefix engine; Python/Node on CPU, one voice, intelligibility only via Whisper round-trip)
- PyTorch rebuild of the backbone loads the k2-fsa checkpoint with no missing/unexpected keys and matches the ct03 logits
  within int8 noise; own KV Qwen3 matches HF (7.6e-5); step-with-cached-prefix equals the full pass when the cache is fresh.
- Exported KV graph matches PyTorch (≈2e-4 on logit scale 135), including zero-length past.
- Cache-once (no refresh) at 16 and 8 steps, fp32 and 4-bit: exact transcripts on 2 prompts. The browser engine's code run on
  onnxruntime-node (padded shapes, JS embeddings): 3/3 prompts exact at 8 steps, guidance on and off.
- Token-level agreement JS vs Python is 100% for 1 step, 96.6% for 2 steps, lower for 8 (near-tie divergence compounds) —
  hence intelligibility, not token equality, is the check.
- NOT verified: WebGPU execution/speed of the new graph (MatMulNBits + KV I/O + zero-length past in ORT-Web), naturalness or
  speaker similarity vs the exact model, other voices/languages, the HF-hosted file layout (files must be uploaded first).

## What is verified vs not (older engine)
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
