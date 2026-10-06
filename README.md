# OmniVoice client-side VAD

Browser-side voice activity detection to accompany
[onnx-community/OmniVoice-Onnx](https://huggingface.co/onnx-community/OmniVoice-Onnx).

- VAD: [Silero VAD](https://github.com/snakers4/silero-vad) through
  [`@ricky0123/vad-web`](https://www.npmjs.com/package/@ricky0123/vad-web) (ONNX Runtime Web, WASM).
  All inference is in the browser; the server only serves static files.
- The page captures mic speech segments and exports them as WAV (16 kHz and 24 kHz), e.g. as a trimmed
  `--ref_audio` clip for OmniVoice voice cloning.
- Not verified: what sample rate the OmniVoice/Higgs encoders expect for reference audio (the model card
  only states 24 kHz *output*). Check `inference.py` / `higgs_inference.py` in the model repo.
- This repo does not run OmniVoice itself.

## Run locally

```bash
npm install     # postinstall copies VAD/ORT assets into public/vendor
npm start       # http://localhost:3000
```

Microphone access needs HTTPS or localhost (Render provides HTTPS).

## Deploy on Render

`render.yaml` defines a Node web service (`npm ci` → `npm start`, health check `/healthz`).
In Render: New → Blueprint → select this repo.
