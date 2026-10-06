// On-the-fly transcription of the selected voice's sample: Whisper via transformers.js on WebGPU.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

export async function loadAsr({ model = "onnx-community/whisper-base", onProgress } = {}) {
  const device = ("gpu" in navigator) ? "webgpu" : "wasm";
  const asr = await pipeline("automatic-speech-recognition", model, {
    device,
    dtype: device === "webgpu" ? { encoder_model: "fp32", decoder_model_merged: "q4" } : "q8",
    progress_callback: onProgress,
  });
  // audio16k: mono Float32Array @ 16 kHz, <= 30 s. Language is left unset so the model detects it.
  return async (audio16k) => ((await asr(audio16k, { task: "transcribe" })).text || "").trim();
}
