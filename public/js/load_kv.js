// Loader for the cached-prefix OmniVoice engine (omnivoice_kv.js) on WebGPU.
// KV_BASE must serve: lm_kv_q4.onnx, lm_kv_q4.onnx.data, embed_text_int8.bin, embed_text_scale.bin, embed_audio.bin
// (produced by export/build_all.sh). The codec (encoder/decoder) and tokenizer come from the ct03 export.
import * as ort from "/vendor/ort/ort.webgpu.min.mjs";
import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { fetchBytes } from "./cache.js";
import { OmniVoiceKV } from "./omnivoice_kv.js";

ort.env.wasm.wasmPaths = "/vendor/ort/";
// Same-origin by default: the files are committed as <=80 MB parts (public/models/kv + manifest.json) because git hosts reject
// bigger files. Any base URL that serves the whole files directly (e.g. a Hugging Face repo) also works, via ?kv=<base>.
export const DEFAULT_KV_BASE = "/models/kv";
const CODEC = "https://huggingface.co/ct03/omnivoice-onnx-int8hq/resolve/main";

// Download a KV-model file; if the base has a manifest.json, fetch its parts in parallel and stitch them together.
let manifestP = null;
async function kvBytes(base, name, onProgress) {
  manifestP ??= fetch(`${base}/manifest.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const man = await manifestP;
  if (!man) return fetchBytes(`${base}/${name}`, onProgress);
  const e = man.files?.[name];
  if (!e) throw new Error(`${name} missing from ${base}/manifest.json`);
  const got = e.parts.map(() => 0), report = () => onProgress?.(got.reduce((a, b) => a + b, 0), e.size);
  const parts = await Promise.all(e.parts.map((p, i) => fetchBytes(`${base}/${p}`, (g) => { got[i] = g; report(); })));
  const out = new Uint8Array(e.size); let off = 0;
  for (const b of parts) { out.set(b, off); off += b.length; }
  if (off !== e.size) throw new Error(`${name}: expected ${e.size} bytes, got ${off}`);
  return out;
}

export async function loadKV({ kvBase = DEFAULT_KV_BASE, onStatus = () => {} } = {}) {
  const gpu = "gpu" in navigator;
  const providers = gpu ? ["webgpu", "wasm"] : ["wasm"];
  const bytes = (label, url) => fetchBytes(url, (g, t) => onStatus(label, g, t));
  const session = async (label, modelUrl, dataUrl, dataPath, extra = {}) => {
    const model = await bytes(label, modelUrl);
    const data = await bytes(label, dataUrl);
    onStatus(label, 1, 1, "initialising");
    return ort.InferenceSession.create(model, { executionProviders: providers, externalData: [{ path: dataPath, data }], ...extra });
  };
  const post = async (n) => ort.InferenceSession.create(new Uint8Array(await (await fetch(`/models/${n}`)).arrayBuffer()), { executionProviders: providers });

  const tok = await AutoTokenizer.from_pretrained("ct03/omnivoice-onnx-int8hq");
  const tf = async (label, name, T) => { const b = await kvBytes(kvBase, name, (g, t) => onStatus(label, g, t)); return new T(b.buffer, b.byteOffset, b.byteLength / T.BYTES_PER_ELEMENT); };
  const tables = { txt: await tf("text embeddings", "embed_text_int8.bin", Int8Array), txtScale: await tf("text scales", "embed_text_scale.bin", Float32Array),
                   aud: await tf("audio embeddings", "embed_audio.bin", Float32Array) };
  // LM: try the hand-fused low-dispatch graph first (shares the same 4-bit weight file), fall back to the traced graph.
  const lmData = await kvBytes(kvBase, "lm_kv_q4.onnx.data", (g, t) => onStatus("OmniVoice LM (4-bit)", g, t));
  const lmSession = async (file) => ort.InferenceSession.create(await kvBytes(kvBase, file), { executionProviders: providers,
    externalData: [{ path: "lm_kv_q4.onnx.data", data: lmData }], ...(gpu ? { preferredOutputLocation: "gpu-buffer" } : {}) });   // logits + K/V stay on the GPU
  onStatus("OmniVoice LM (4-bit)", 1, 1, "initialising");
  const decoder = await session("codec decoder", `${CODEC}/audio_tokenizer_decoder_int8/model.onnx`, `${CODEC}/audio_tokenizer_decoder_int8/model.onnx_data`, "model.onnx_data");
  const encoder = await session("codec encoder", `${CODEC}/audio_tokenizer_encoder_int8/model.onnx`, `${CODEC}/audio_tokenizer_encoder_int8/model.onnx_data`, "model.onnx_data");
  const postCfg = await post("post_cfg.onnx"), postNoCfg = await post("post_nocfg.onnx");
  const make = (lm) => new OmniVoiceKV({ ort, gpu, providers, tables, tokenizer: { encode: (t) => tok.encode(t, { add_special_tokens: false }) },
    sessions: { lm, decoder, encoder, postCfg, postNoCfg } });
  // Warm-up: the first run of each kernel/shape pays a one-off compile cost (~seconds on WebGPU); pay it here, not on the first Speak.
  const warm = async (engine) => {
    const frames = 200, codes = Int32Array.from({ length: 8 * frames }, (_, i) => (i * 37) % 1024);
    await engine.synthesize("This is a short warm up sentence.", { codes, frames, rms: 0.1, text: "Warm up text for the reference." }, { numStep: 2, seed: 1 });
  };
  let engine = null;
  for (const file of ["lm_fused.onnx", "lm_kv_q4.onnx"]) {
    try {
      const lm = await lmSession(file);
      onStatus("GPU warm-up", 0, 0, `compiling kernels (${file})…`);
      engine = make(lm); await warm(engine);
      onStatus("GPU warm-up", 1, 1, "done");
      break;
    } catch (e) { console.warn(`${file} failed, trying the next graph:`, e); engine = null; }
  }
  if (!engine) throw new Error("no LM graph could be run on this device");
  return engine;
}
