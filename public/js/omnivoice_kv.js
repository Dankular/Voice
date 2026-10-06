// OmniVoice with a cached prefix: one full LM pass computes K/V of [style | text | voice sample]; each later step runs
// only the target positions against that cache (≈ target/total of the FLOPs). The LM is the single KV-exposing graph from
// export/export_kv.py (4-bit MatMulNBits). Embedding lookups are done here from small tables. Env-agnostic core:
// `ort` is injected so the same code runs on WebGPU (browser) and onnxruntime-node (tests).
import { estimateDuration } from "./duration.js";
import { rms } from "./audio.js";

export const SAMPLE_RATE = 24000;
const HOP = 960, MASK = 1024, NC = 8, H = 1024, NL = 28, AV = 1025;
const NONVERBAL = /\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]/g;

const bucket = (n, m) => Math.ceil(n / m) * m;
function rng(seed) {
  let a = (seed ?? (Math.random() * 2 ** 32)) >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const combineText = (text, refText) => {
  let t = refText ? refText.trim() + " " + text.trim() : text.trim();
  t = t.replace(/[\r\n]+/g, "").replace(/（/g, "(").replace(/）/g, ")").replace(/[ \t]+/g, " ");
  return t.replace(/(?<=[一-鿿])\s+|\s+(?=[一-鿿])/g, "");
};
// real positions [0,L) attend to each other; padding rows attend only to themselves (keeps rows non-empty)
function blockMask(L, S) {
  const m = new Uint8Array(S * S);
  for (let r = 0; r < L; r++) m.fill(1, r * S, r * S + L);
  for (let r = L; r < S; r++) m[r * S + r] = 1;
  return m;
}

// fused graph takes an additive float bias [1,1,2Q,L] (the Q x L mask, 0 / -1e9, duplicated for its 2 query groups per KV head)
function biasFrom(mask, Q, Lc) {
  const b = new Float32Array(2 * Q * Lc);
  for (let r = 0; r < Q; r++) for (let c = 0; c < Lc; c++) { const v = mask[r * Lc + c] ? 0 : -1e9; b[r * Lc + c] = v; b[(Q + r) * Lc + c] = v; }
  return b;
}

export class OmniVoiceKV {
  /** sessions: {lm, decoder, encoder, postCfg, postNoCfg}; tables: {txt:Int8Array, txtScale:Float32Array, aud:Float32Array}
   *  tokenizer: {encode(text)->ids}; gpu: keep LM outputs on the GPU (browser WebGPU only). */
  constructor({ ort, sessions, tables, tokenizer, gpu = false, providers = ["wasm"] }) {
    Object.assign(this, { ort, ...sessions, tables, tok: tokenizer, gpu, providers });
    this.fused = this.lm.inputNames.includes("attention_bias");   // hand-fused low-dispatch graph (see export/build_fused.py)
    this.maskVec = new Float32Array(H);
    this.audioSum(new Int32Array(NC).fill(MASK), 0, this.maskVec, 0);
  }

  ids(text) { return Array.from(this.tok.encode(text)); }
  textIds(text) {
    const out = []; let last = 0;
    for (const m of text.matchAll(NONVERBAL)) {
      if (m.index > last) out.push(...this.ids(text.slice(last, m.index)));
      out.push(...this.ids(m[0])); last = m.index + m[0].length;
    }
    if (last < text.length) out.push(...this.ids(text.slice(last)));
    return out;
  }

  // ---- embeddings (the graph takes inputs_embeds) ----
  textRow(id, out, o) { const { txt, txtScale } = this.tables, s = txtScale[id], b = id * H; for (let d = 0; d < H; d++) out[o + d] = txt[b + d] * s; }
  // codes: Int32Array; reads codes[c*stride + pos] for c<8
  audioSum(codes, pos, out, o, stride = 1) {
    const { aud } = this.tables;
    for (let d = 0; d < H; d++) out[o + d] = 0;
    for (let c = 0; c < NC; c++) { const b = (codes[c * stride + pos] + c * AV) * H; for (let d = 0; d < H; d++) out[o + d] += aud[b + d]; }
  }

  async encodeReference(wav24) {
    const r = rms(wav24);
    let x = wav24;
    if (r > 0 && r < 0.1) x = wav24.map((v) => v * 0.1 / r);
    x = x.subarray(0, Math.floor(x.length / HOP) * HOP);   // codec hop is 960 samples (25 fps); other lengths break the encoder
    const out = await this.encoder.run({ audio: new this.ort.Tensor("float32", x, [1, 1, x.length]) });
    const t = out.audio_codes;
    return { codes: Int32Array.from(t.data, Number), frames: t.dims[2], rms: r };
  }

  /** Speak `text` in the cloned voice. ref = {codes, frames, rms, text}. Returns Float32Array @ 24 kHz
   *  (with .timing, and .tokens when opts.returnTokens). */
  async synthesize(text, ref, { language = null, numStep = 8, guidance = 2.0, tShift = 0.1, layerPenalty = 5.0,
                                positionTemp = 5.0, speed = 1.0, seed, refresh = 0, onStep, returnTokens = false } = {}) {
    const { ort } = this;
    const Tr = ref.frames;
    let T = Math.max(1, Math.floor(estimateDuration(text, ref.text, Tr)));
    if (speed > 0 && speed !== 1) T = Math.max(1, Math.floor(T / speed));

    const style = "<|denoise|>" + `<|lang_start|>${language || "None"}<|lang_end|><|instruct_start|>None<|instruct_end|>`;
    const prefix = [...this.ids(style), ...this.textIds(`<|text_start|>${combineText(text, ref.text)}<|text_end|>`)];
    const P = prefix.length, t0 = P + Tr, Lc = t0 + T;
    const Tu = bucket(T, 32);                      // target slots per pass (>= T); shapes bucketed so GPU kernels are reused
    const Sc = bucket(t0 + Tu, 64);                // full-pass length; padding is masked out of attention

    // ---- embeddings ----
    const fullEmb = new Float32Array(Sc * H), tgtEmb = new Float32Array(Tu * H);
    for (let i = 0; i < P; i++) this.textRow(prefix[i], fullEmb, i * H);
    for (let t = 0; t < Tr; t++) this.audioSum(ref.codes, t, fullEmb, (P + t) * H, Tr);
    for (let t = 0; t < Tu; t++) {
      if (t < T) { fullEmb.set(this.maskVec, (t0 + t) * H); tgtEmb.set(this.maskVec, t * H); }
      else this.textRow(MASK, tgtEmb, t * H);      // padding rows: any finite values (masked)
    }
    for (let i = t0 + T; i < Sc; i++) this.textRow(MASK, fullEmb, i * H);

    // ---- constant inputs ----
    const T_ = (type, data, dims) => new ort.Tensor(type, data, dims);
    const emptyV = T_("float32", new Float32Array(0), [1, 8, 0, 128]);
    const emptyK = this.fused ? T_("float32", new Float32Array(0), [1, 8, 128, 0]) : emptyV;     // fused graph keeps K transposed
    const noPast = {}; for (let i = 0; i < NL; i++) { noPast[`past_key_values.${i}.key`] = emptyK; noPast[`past_key_values.${i}.value`] = emptyV; }
    const posFull = BigInt64Array.from({ length: Sc }, (_, i) => BigInt(i));
    const posUnc = BigInt64Array.from({ length: Tu }, (_, i) => BigInt(i));
    const posStep = BigInt64Array.from({ length: Tu }, (_, i) => BigInt(t0 + i));
    const maskT = (m, Q, Lcols) => this.fused ? { attention_bias: T_("float32", biasFrom(m, Q, Lcols), [1, 1, 2 * Q, Lcols]) }
                                                : { attention_mask: T_("bool", m, [1, 1, Q, Lcols]) };
    const fullMask = maskT(blockMask(Lc, Sc), Sc, Sc);
    const uncMask = maskT(blockMask(T, Tu), Tu, Tu);
    const stepMaskArr = new Uint8Array(Tu * (Sc + Tu));                      // [Tu, Sc+Tu]
    for (let r = 0; r < T; r++) { stepMaskArr.fill(1, r * (Sc + Tu), r * (Sc + Tu) + t0); stepMaskArr.fill(1, r * (Sc + Tu) + Sc, r * (Sc + Tu) + Sc + T); }
    for (let r = T; r < Tu; r++) stepMaskArr[r * (Sc + Tu) + Sc + r] = 1;    // pads: diagonal only
    const stepMask = maskT(stepMaskArr, Tu, Sc + Tu);

    const post = guidance !== 0 ? this.postCfg : this.postNoCfg;
    const t0T = (v) => T_("int64", BigInt64Array.of(BigInt(v)), [1]);
    const gT = T_("float32", Float32Array.of(guidance), []);
    const tlenT = T_("int64", BigInt64Array.of(BigInt(Tu)), [1]);

    // ---- schedule (upstream _get_time_steps) ----
    const ts = Array.from({ length: numStep + 1 }, (_, i) => tShift * (i / numStep) / (1 + (tShift - 1) * (i / numStep)));
    const total = T * NC; let rem = total; const sched = [];
    for (let s = 0; s < numStep; s++) { const k = s === numStep - 1 ? rem : Math.min(Math.ceil(total * (ts[s + 1] - ts[s])), rem); sched.push(k); rem -= k; }

    const rand = rng(seed);
    const tokens = new Int32Array(NC * T).fill(MASK);
    const timing = { steps: [], shapes: { full: Sc, step: Tu, T } };
    let past = null;
    const disposePast = () => { if (past) for (const k in past) past[k]?.dispose?.(); past = null; };

    for (let s = 0; s < numStep; s++) {
      const k = sched[s];
      if (k <= 0) continue;
      const m0 = performance.now();
      let cLog, condT0, isFull = false;
      if (s === 0 || (refresh > 0 && s % refresh === 0)) {          // full pass: refreshes the prefix K/V cache
        disposePast();
        const out = await this.lm.run({ inputs_embeds: T_("float32", fullEmb, [1, Sc, H]), ...fullMask,
                                        position_ids: T_("int64", posFull, [1, Sc]), ...noPast });
        cLog = out.logits; condT0 = t0; isFull = true; past = {};
        for (let i = 0; i < NL; i++) for (const kv of ["key", "value"]) past[`past_key_values.${i}.${kv}`] = out[`present.${i}.${kv}`];
      } else {                                                       // target-only pass against the cached prefix
        cLog = (await this.lm.run({ inputs_embeds: T_("float32", tgtEmb, [1, Tu, H]), ...stepMask,
                                    position_ids: T_("int64", posStep, [1, Tu]), ...past }, ["logits"])).logits;
        condT0 = 0;
      }
      let uLog = null;
      if (guidance !== 0) uLog = (await this.lm.run({ inputs_embeds: T_("float32", tgtEmb, [1, Tu, H]), ...uncMask,
                                                      position_ids: T_("int64", posUnc, [1, Tu]), ...noPast }, ["logits"])).logits;
      const m1 = performance.now();
      const o = await post.run(uLog ? { cond: cLog, uncond: uLog, t0: t0T(condT0), g: gT } : { cond: cLog, t0: t0T(condT0), tlen: tlenT });
      cLog.dispose?.(); uLog?.dispose?.();
      const pred = o.pred.data, conf = o.conf.data;                  // [1,8,Tu]
      const m2 = performance.now();

      const cand = [];
      for (let c = 0; c < NC; c++) for (let t = 0; t < T; t++) {
        const idx = c * T + t;
        if (tokens[idx] !== MASK) continue;
        let score = conf[c * Tu + t] - c * layerPenalty;
        if (positionTemp > 0) score = score / positionTemp - Math.log(-Math.log(rand() + 1e-10) + 1e-10);
        cand.push([score, idx, Number(pred[c * Tu + t])]);
      }
      cand.sort((a, b) => b[0] - a[0]);
      const changed = new Set();
      for (let j = 0; j < Math.min(k, cand.length); j++) { const [, idx, tok] = cand[j]; tokens[idx] = tok; changed.add(idx % T); }
      for (const t of changed) {                                     // refresh embeddings of the positions that changed
        const col = new Int32Array(NC); for (let c = 0; c < NC; c++) col[c] = tokens[c * T + t];
        this.audioSum(col, 0, tgtEmb, t * H); fullEmb.set(tgtEmb.subarray(t * H, (t + 1) * H), (t0 + t) * H);
      }
      const info = { step: s + 1, of: numStep, lmMs: m1 - m0, postMs: m2 - m1, jsMs: performance.now() - m2, full: isFull };
      timing.steps.push(info); onStep?.(info);
    }
    disposePast();

    const out = await this.decoder.run({ audio_codes: new ort.Tensor("int64", BigInt64Array.from(tokens, BigInt), [1, NC, T]) });
    const wav = postProcess(Float32Array.from(out.audio.data), ref.rms);
    wav.timing = timing; if (returnTokens) wav.tokens = tokens;
    return wav;
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
