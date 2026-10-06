// OmniVoice in the browser: ONNX Runtime Web (WebGPU, wasm fallback) + a JS port of upstream's decoding loop
// (k2-fsa/OmniVoice omnivoice/models/omnivoice.py, Apache-2.0). Voice cloning only.
import * as ort from "/vendor/ort/ort.webgpu.min.mjs";
import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { fetchBytes } from "./cache.js";
import { estimateDuration } from "./duration.js";
import { rms } from "./audio.js";

ort.env.wasm.wasmPaths = "/vendor/ort/";

export const SAMPLE_RATE = 24000;
const HOP = 960, MASK = 1024, NC = 8;
const NONVERBAL = /\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]/g;

function rng(seed) {   // mulberry32
  let a = (seed ?? (Math.random() * 2 ** 32)) >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const combineText = (text, refText) => {
  let t = refText ? refText.trim() + " " + text.trim() : text.trim();
  t = t.replace(/[\r\n]+/g, "").replace(/（/g, "(").replace(/）/g, ")").replace(/[ \t]+/g, " ");
  return t.replace(/(?<=[一-鿿])\s+|\s+(?=[一-鿿])/g, "");
};

export class OmniVoice {
  static async load({ repo = "ct03/omnivoice-onnx-int8hq", onStatus = () => {} } = {}) {
    const base = `https://huggingface.co/${repo}/resolve/main`;
    const providers = ("gpu" in navigator) ? ["webgpu", "wasm"] : ["wasm"];
    const gpu = providers[0] === "webgpu";
    const mk = async (label, dir, extra = {}) => {
      const model = await fetchBytes(`${base}/${dir}/model.onnx`);
      const data = await fetchBytes(`${base}/${dir}/model.onnx_data`, (g, t) => onStatus(label, g, t));
      onStatus(label, 1, 1, "initialising");
      return ort.InferenceSession.create(model, { executionProviders: providers, externalData: [{ path: "model.onnx_data", data }], ...extra });
    };
    const post = async (name) => ort.InferenceSession.create(new Uint8Array(await (await fetch(`/models/${name}`)).arrayBuffer()),
      { executionProviders: providers });
    const o = new OmniVoice();
    o.providers = providers;
    o.tok = await AutoTokenizer.from_pretrained(repo);
    // keep the logits on the GPU: they go straight into the post-processing graph, only its small output is read back
    o.lm = await mk("OmniVoice LM", "omnivoice_lm_int8_hq", gpu ? { preferredOutputLocation: { logits: "gpu-buffer" } } : {});
    o.postCfg = await post("post_cfg.onnx");
    o.postNoCfg = await post("post_nocfg.onnx");
    o.decoder = await mk("codec decoder", "audio_tokenizer_decoder_int8");
    o.encoder = await mk("codec encoder", "audio_tokenizer_encoder_int8");
    return o;
  }

  ids(text) { return Array.from(this.tok.encode(text, { add_special_tokens: false })); }

  textIds(text) {   // non-verbal tags are tokenised standalone, as upstream does
    const out = []; let last = 0;
    for (const m of text.matchAll(NONVERBAL)) {
      if (m.index > last) out.push(...this.ids(text.slice(last, m.index)));
      out.push(...this.ids(m[0])); last = m.index + m[0].length;
    }
    if (last < text.length) out.push(...this.ids(text.slice(last)));
    return out;
  }

  /** wav24: mono Float32Array at 24 kHz -> {codes: Int32Array(8*Tr) row-major, frames, rms} */
  async encodeReference(wav24) {
    const r = rms(wav24);
    let x = wav24;
    if (r > 0 && r < 0.1) { x = wav24.map((v) => v * 0.1 / r); }
    x = x.subarray(0, Math.floor(x.length / HOP) * HOP);   // codec hop is 960 (25 fps); other lengths break the encoder
    const out = await this.encoder.run({ audio: new ort.Tensor("float32", x, [1, 1, x.length]) });
    const t = out.audio_codes;
    return { codes: Int32Array.from(t.data, Number), frames: t.dims[2], rms: r };
  }

  /** Speak `text` in the cloned voice. ref = {codes, frames, rms, text}. Returns Float32Array @ 24 kHz.
   *  opts: numStep, guidance (0 disables the unconditional pass: ~2x faster), speed, seed, onStep(info). */
  async synthesize(text, ref, { language = null, numStep = 32, guidance = 2.0, tShift = 0.1, layerPenalty = 5.0,
                                positionTemp = 5.0, speed = 1.0, seed, onStep } = {}) {
    const Tr = ref.frames;
    let T = Math.max(1, Math.floor(estimateDuration(text, ref.text, Tr)));
    if (speed > 0 && speed !== 1) T = Math.max(1, Math.floor(T / speed));

    const style = "<|denoise|>" + `<|lang_start|>${language || "None"}<|lang_end|><|instruct_start|>None<|instruct_end|>`;
    const prefix = [...this.ids(style), ...this.textIds(`<|text_start|>${combineText(text, ref.text)}<|text_end|>`)];
    const P = prefix.length, t0 = P + Tr;

    // Shapes are bucketed so the WebGPU backend can reuse the kernels it compiled for earlier utterances. Padding
    // positions are masked out of attention (4D mask), so real positions see exactly what they would unpadded.
    const Tu = bucket(T, 32);                 // target slots read back per step (>= T)
    const Lc = t0 + T, Sc = bucket(t0 + Tu, 64);

    const condIds = new BigInt64Array(NC * Sc).fill(BigInt(MASK));
    for (let c = 0; c < NC; c++) {
      for (let i = 0; i < P; i++) condIds[c * Sc + i] = BigInt(prefix[i]);
      for (let t = 0; t < Tr; t++) condIds[c * Sc + P + t] = BigInt(ref.codes[c * Tr + t]);
    }
    const condAudio = new Uint8Array(Sc); condAudio.fill(1, P, Lc);    // reference + target are audio positions
    const uncIds = new BigInt64Array(NC * Tu).fill(BigInt(MASK));      // unconditional branch: target only
    const uncAudio = new Uint8Array(Tu); uncAudio.fill(1, 0, T);

    const feeds = (ids, audio, L, S) => ({
      input_ids: new ort.Tensor("int64", ids, [1, NC, S]),
      audio_mask: new ort.Tensor("bool", audio, [1, S]),
      attention_mask: new ort.Tensor("bool", blockMask(L, S), [1, 1, S, S]),
      position_ids: new ort.Tensor("int64", positions(S), [1, S]),
    });
    const cf = feeds(condIds, condAudio, Lc, Sc);                      // built once; input_ids is updated in place
    const uf = guidance !== 0 ? feeds(uncIds, uncAudio, T, Tu) : null;
    const post = guidance !== 0 ? this.postCfg : this.postNoCfg;
    const t0T = new ort.Tensor("int64", BigInt64Array.of(BigInt(t0)), [1]);
    const extra = guidance !== 0
      ? { t0: t0T, g: new ort.Tensor("float32", Float32Array.of(guidance), []) }
      : { t0: t0T, tlen: new ort.Tensor("int64", BigInt64Array.of(BigInt(Tu)), [1]) };

    // unmasking schedule (upstream _get_time_steps)
    const ts = Array.from({ length: numStep + 1 }, (_, i) => tShift * (i / numStep) / (1 + (tShift - 1) * (i / numStep)));
    const total = T * NC; let rem = total; const sched = [];
    for (let s = 0; s < numStep; s++) {
      const k = s === numStep - 1 ? rem : Math.min(Math.ceil(total * (ts[s + 1] - ts[s])), rem);
      sched.push(k); rem -= k;
    }

    const rand = rng(seed);
    const tokens = new Int32Array(NC * T).fill(MASK);
    const timing = { steps: [], shapes: { cond: Sc, uncond: guidance !== 0 ? Tu : 0, T } };

    for (let s = 0; s < numStep; s++) {
      const k = sched[s];
      if (k <= 0) continue;
      const m0 = performance.now();
      const cLog = (await this.lm.run(cf)).logits;                     // GPU-resident [1,8,Sc,1025]
      const uLog = uf ? (await this.lm.run(uf)).logits : null;
      const m1 = performance.now();
      const o = await post.run(uLog ? { cond: cLog, uncond: uLog, ...extra } : { cond: cLog, ...extra });
      cLog.dispose?.(); uLog?.dispose?.();
      const pred = o.pred.data, conf = o.conf.data;                    // [1,8,Tu] each: tiny readback
      const m2 = performance.now();

      const cand = [];                                                 // [score, flatIndex, predToken]
      for (let c = 0; c < NC; c++) for (let t = 0; t < T; t++) {
        const idx = c * T + t;
        if (tokens[idx] !== MASK) continue;                            // already unmasked: can't be picked
        let score = conf[c * Tu + t] - c * layerPenalty;
        if (positionTemp > 0) score = score / positionTemp - Math.log(-Math.log(rand() + 1e-10) + 1e-10);
        cand.push([score, idx, Number(pred[c * Tu + t])]);
      }
      cand.sort((x, y) => y[0] - x[0]);
      for (let j = 0; j < Math.min(k, cand.length); j++) {
        const [, idx, tok] = cand[j], c = Math.floor(idx / T), t = idx % T;
        tokens[idx] = tok;
        condIds[c * Sc + t0 + t] = BigInt(tok); uncIds[c * Tu + t] = BigInt(tok);
      }
      const m3 = performance.now();
      const info = { step: s + 1, of: numStep, lmMs: m1 - m0, postMs: m2 - m1, jsMs: m3 - m2 };
      timing.steps.push(info);
      onStep?.(info);
    }

    const out = await this.decoder.run({ audio_codes: new ort.Tensor("int64", BigInt64Array.from(tokens, BigInt), [1, NC, T]) });
    const wav = postProcess(Float32Array.from(out.audio.data), ref.rms);
    wav.timing = timing;
    return wav;
  }
}

const bucket = (n, m) => Math.ceil(n / m) * m;
const posCache = new Map();
const positions = (S) => posCache.get(S) ?? (posCache.set(S, BigInt64Array.from({ length: S }, (_, i) => BigInt(i))), posCache.get(S));
// real positions [0,L) attend to each other; padding positions attend only to themselves (keeps rows non-empty)
function blockMask(L, S) {
  const m = new Uint8Array(S * S);
  for (let r = 0; r < L; r++) m.fill(1, r * S, r * S + L);
  for (let r = L; r < S; r++) m[r * S + r] = 1;
  return m;
}

// Upstream post-processing minus silence removal: match quiet references' loudness, then 0.1 s fade + pad.
function postProcess(w, refRms) {
  if (refRms != null && refRms < 0.1) { const g = refRms / 0.1; for (let i = 0; i < w.length; i++) w[i] *= g; }
  const k = Math.min(Math.floor(0.1 * SAMPLE_RATE), w.length >> 1), pad = Math.floor(0.1 * SAMPLE_RATE);
  for (let i = 0; i < k; i++) { w[i] *= i / k; w[w.length - 1 - i] *= i / k; }
  const out = new Float32Array(w.length + 2 * pad); out.set(w, pad);
  return out;
}
