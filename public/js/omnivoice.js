// OmniVoice in the browser: ONNX Runtime Web (WebGPU, wasm fallback) + a JS port of upstream's decoding loop
// (k2-fsa/OmniVoice omnivoice/models/omnivoice.py, Apache-2.0). Voice cloning only.
import * as ort from "/vendor/ort/ort.webgpu.min.mjs";
import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { fetchBytes } from "./cache.js";
import { estimateDuration } from "./duration.js";
import { rms } from "./audio.js";

ort.env.wasm.wasmPaths = "/vendor/ort/";

export const SAMPLE_RATE = 24000;
const HOP = 960, MASK = 1024, NC = 8, VOCAB = 1025;
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

function logSoftmax(src, off, n, dst) {   // dst may alias src
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (src[off + i] > m) m = src[off + i];
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.exp(src[off + i] - m);
  const l = m + Math.log(s);
  for (let i = 0; i < n; i++) dst[i] = src[off + i] - l;
}

export class OmniVoice {
  static async load({ repo = "ct03/omnivoice-onnx-int8hq", onStatus = () => {} } = {}) {
    const base = `https://huggingface.co/${repo}/resolve/main`;
    const providers = ("gpu" in navigator) ? ["webgpu", "wasm"] : ["wasm"];
    const mk = async (label, dir) => {
      const model = await fetchBytes(`${base}/${dir}/model.onnx`);
      const data = await fetchBytes(`${base}/${dir}/model.onnx_data`, (g, t) => onStatus(label, g, t));
      onStatus(label, 1, 1, "initialising");
      return ort.InferenceSession.create(model, { executionProviders: providers, externalData: [{ path: "model.onnx_data", data }] });
    };
    const o = new OmniVoice();
    o.providers = providers;
    o.tok = await AutoTokenizer.from_pretrained(repo);
    o.lm = await mk("OmniVoice LM", "omnivoice_lm_int8_hq");
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

  async forward(ids, audioMask, S) {
    const out = await this.lm.run({
      input_ids: new ort.Tensor("int64", ids, [1, NC, S]),
      audio_mask: new ort.Tensor("bool", audioMask, [1, S]),
      attention_mask: new ort.Tensor("bool", new Uint8Array(S * S).fill(1), [1, 1, S, S]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from({ length: S }, (_, i) => BigInt(i)), [1, S]),
    });
    return out.logits.data;   // Float32Array [1, 8, S, 1025]
  }

  /** Speak `text` in the cloned voice. ref = {codes, frames, rms, text}. Returns Float32Array @ 24 kHz. */
  async synthesize(text, ref, { language = null, numStep = 32, guidance = 2.0, tShift = 0.1, layerPenalty = 5.0,
                                positionTemp = 5.0, speed = 1.0, seed, onStep } = {}) {
    const Tr = ref.frames;
    let T = Math.max(1, Math.floor(estimateDuration(text, ref.text, Tr)));
    if (speed > 0 && speed !== 1) T = Math.max(1, Math.floor(T / speed));

    const style = "<|denoise|>" + `<|lang_start|>${language || "None"}<|lang_end|><|instruct_start|>None<|instruct_end|>`;
    const prefix = [...this.ids(style), ...this.textIds(`<|text_start|>${combineText(text, ref.text)}<|text_end|>`)];
    const P = prefix.length, t0 = P + Tr, S = t0 + T;

    // cond ids [8, S]: text prefix (same id on every codebook) | reference codes | MASK target
    const cond = new BigInt64Array(NC * S).fill(BigInt(MASK));
    for (let c = 0; c < NC; c++) {
      for (let i = 0; i < P; i++) cond[c * S + i] = BigInt(prefix[i]);
      for (let t = 0; t < Tr; t++) cond[c * S + P + t] = BigInt(ref.codes[c * Tr + t]);
    }
    const condMask = new Uint8Array(S); condMask.fill(1, P);             // reference + target are audio positions
    const unc = new BigInt64Array(NC * T).fill(BigInt(MASK));            // unconditional branch: target only
    const uncMask = new Uint8Array(T).fill(1);

    // unmasking schedule (upstream _get_time_steps)
    const ts = Array.from({ length: numStep + 1 }, (_, i) => tShift * (i / numStep) / (1 + (tShift - 1) * (i / numStep)));
    const total = T * NC; let rem = total; const sched = [];
    for (let s = 0; s < numStep; s++) {
      const k = s === numStep - 1 ? rem : Math.min(Math.ceil(total * (ts[s + 1] - ts[s])), rem);
      sched.push(k); rem -= k;
    }

    const rand = rng(seed);
    const tokens = new Int32Array(NC * T).fill(MASK);
    const cl = new Float32Array(VOCAB), ul = new Float32Array(VOCAB), mix = new Float32Array(VOCAB);

    for (let s = 0; s < numStep; s++) {
      const k = sched[s];
      if (k <= 0) continue;
      const cLog = await this.forward(cond, condMask, S);
      const uLog = guidance !== 0 ? await this.forward(unc, uncMask, T) : null;
      const cand = [];                                                   // [score, flatIndex, predToken]
      for (let c = 0; c < NC; c++) for (let t = 0; t < T; t++) {
        const idx = c * T + t;
        if (tokens[idx] !== MASK) continue;                              // already unmasked: can't be picked
        logSoftmax(cLog, (c * S + t0 + t) * VOCAB, VOCAB, cl);
        if (uLog) {
          logSoftmax(uLog, (c * T + t) * VOCAB, VOCAB, ul);
          for (let v = 0; v < VOCAB; v++) mix[v] = cl[v] + guidance * (cl[v] - ul[v]);
          logSoftmax(mix, 0, VOCAB, mix);
        } else mix.set(cl);
        let best = -Infinity, arg = 0;
        for (let v = 0; v < MASK; v++) if (mix[v] > best) { best = mix[v]; arg = v; }   // mask id is excluded
        let score = best - c * layerPenalty;
        if (positionTemp > 0) score = score / positionTemp - Math.log(-Math.log(rand() + 1e-10) + 1e-10);
        cand.push([score, idx, arg]);
      }
      cand.sort((a, b) => b[0] - a[0]);
      for (let j = 0; j < Math.min(k, cand.length); j++) {
        const [, idx, tok] = cand[j];
        tokens[idx] = tok;
        const c = Math.floor(idx / T), t = idx % T;
        cond[c * S + t0 + t] = BigInt(tok); unc[c * T + t] = BigInt(tok);
      }
      onStep?.(s + 1, numStep);
    }

    const out = await this.decoder.run({ audio_codes: new ort.Tensor("int64", BigInt64Array.from(tokens, BigInt), [1, NC, T]) });
    return postProcess(Float32Array.from(out.audio.data), ref.rms);
  }
}

// Upstream post-processing minus silence removal: match quiet references' loudness, then 0.1 s fade + pad.
function postProcess(w, refRms) {
  if (refRms != null && refRms < 0.1) { const g = refRms / 0.1; for (let i = 0; i < w.length; i++) w[i] *= g; }
  const k = Math.min(Math.floor(0.1 * SAMPLE_RATE), w.length >> 1), pad = Math.floor(0.1 * SAMPLE_RATE);
  for (let i = 0; i < k; i++) { w[i] *= i / k; w[w.length - 1 - i] *= i / k; }
  const out = new Float32Array(w.length + 2 * pad); out.set(w, pad);
  return out;
}
