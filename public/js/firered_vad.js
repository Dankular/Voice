// FireRedVAD Stream-VAD (DFSMN, ONNX): speech probability per 10 ms frame from streaming 80-dim fbank features.
// Model I/O (from the ONNX graph): feat [1,T,80] + cache_0..7 [1,128,L] -> probs [1,T,1] + new_cache_0..7.
// Caches start as zeros with L=10 and are fed back exactly as returned (their length settles at 19; trimming them changes the output).
import { Fbank } from "./fbank.js";

const BLOCKS = 8, DIM = 128, CACHE0 = 10;

export class FireRedVAD {
  // `ort`: an onnxruntime-web / onnxruntime-node module; modelBytes: the .onnx file; cmvn: parsed cmvn.json ({means, inv_stddevs}).
  static async create(ort, modelBytes, cmvn, sessionOptions = { executionProviders: ["wasm"], graphOptimizationLevel: "all" }) {
    return new FireRedVAD(ort, await ort.InferenceSession.create(modelBytes, sessionOptions), cmvn);
  }
  static async fetch(ort, base = "/models/vad", sessionOptions) {
    const [m, c] = await Promise.all([fetch(`${base}/firered_vad.onnx`).then((r) => r.arrayBuffer()), fetch(`${base}/cmvn.json`).then((r) => r.json())]);
    return FireRedVAD.create(ort, new Uint8Array(m), c, sessionOptions);
  }
  constructor(ort, session, cmvn) {
    this.ort = ort; this.session = session; this.mean = Float32Array.from(cmvn.means); this.inv = Float32Array.from(cmvn.inv_stddevs);
    this.reset();
  }
  reset() {
    this.caches = Array.from({ length: BLOCKS }, () => new this.ort.Tensor("float32", new Float32Array(DIM * CACHE0), [1, DIM, CACHE0]));
  }
  // frames: Float32Array(80)[] of raw log-fbank -> Float32Array of speech probabilities (one per frame)
  async probs(frames) {
    const T = frames.length, feat = new Float32Array(T * 80);
    frames.forEach((f, t) => { for (let k = 0; k < 80; k++) feat[t * 80 + k] = (f[k] - this.mean[k]) * this.inv[k]; });
    const inputs = { feat: new this.ort.Tensor("float32", feat, [1, T, 80]) };
    this.caches.forEach((c, i) => (inputs[`cache_${i}`] = c));
    const out = await this.session.run(inputs);
    this.caches = this.caches.map((_, i) => out[`new_cache_${i}`]);
    return Float32Array.from(out.probs.data);
  }
}

// Samples in (16 kHz float32) -> speech probabilities out, in order. push() calls may overlap; they are processed one at a time.
export class StreamingVad {
  constructor(vad) { this.vad = vad; this.fbank = new Fbank(); this.tail = Promise.resolve(); }
  reset() { this.fbank.reset(); this.vad.reset(); }
  push(samples) {
    const frames = this.fbank.push(samples);
    const run = this.tail.then(() => (frames.length ? this.vad.probs(frames) : new Float32Array(0)));
    this.tail = run.catch(() => {});
    return run;
  }
}
