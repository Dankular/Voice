# OmniVoice voice design + client-side VAD

Web service around [onnx-community/OmniVoice-Onnx](https://huggingface.co/onnx-community/OmniVoice-Onnx):

- **Client-side VAD** (browser): Silero VAD via `@ricky0123/vad-web`; captures speech segments, exports WAV.
- **Voice-design TTS** (server, CPU, ONNX Runtime): filter by gender, age, pitch ("tone"), whisper, English accent,
  Chinese dialect. The vocabulary is the one documented upstream (k2-fsa/OmniVoice `docs/voice-design.md`);
  upstream documents no emotion control, so "tone" = pitch (+ whisper).
- `get_voices.py` (ElevenLabs library dump) gains a `tts` subcommand; existing usage is unchanged.

## KNOWN ISSUE: output is not intelligible with the onnx-community backbone

Measured, not assumed:
- `onnx-community/OmniVoice-Onnx`'s `llm_decoder.onnx` is **causal**: changing only the last input position leaves
  every earlier position's output unchanged (tested on the int4 build). Upstream builds a full-block attention mask,
  so by reading its code the real model attends bidirectionally (not yet confirmed by running upstream PyTorch).
- Whisper-base (ONNX, in `omnivoice_tts/asr.py`, which transcribes real speech correctly) cannot recover the text from
  this engine's output: it returns unrelated phrases / repetition loops.
- So treat the voice-design output here as **unverified/likely broken**, and the pitch/gender filters as untested.
- Other exports (e.g. `ct03/omnivoice-onnx-int8hq`) document a 4D `attention_mask` input; being evaluated.

`omnivoice_tts/asr.py` (Whisper on ONNX Runtime, for transcribing reference audio) is verified on a real speech sample.

Not implemented: voice cloning, upstream's pydub silence removal, long-text chunking. Model licence is CC-BY-NC
(per the k2-fsa/OmniVoice card), which matters for hosting a public service.

## CLI
```bash
pip install -r requirements-tts.txt
./get_voices.py tts --list-attributes
./get_voices.py tts --text "Hello" --gender female --pitch "low pitch" --accent "british accent" -o out.wav
```
Models (~430 MB) download on first use to `$OMNIVOICE_HOME` (default `~/.cache/omnivoice-onnx`).

## Run the service
```bash
npm ci --ignore-scripts && node scripts/vendor.mjs   # browser VAD assets -> public/vendor
uvicorn app:app --port 10000
```
API: `GET /api/attributes`, `POST /api/tts` (JSON -> audio/wav), `GET /healthz`.

## Render
`render.yaml` is a Docker web service with a persistent disk for the model cache. The Docker build and the
Render deploy have **not** been run.
