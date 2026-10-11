// Usage: node scripts/test_vad_stream.mjs <audio.f32> <reference_probs.f32> [chunkSamples]
// Runs fbank -> FireRedVAD (onnxruntime-web, wasm) -> TurnDetector on 16 kHz float32 audio fed in chunks, and compares the
// probabilities with the Python reference (kaldi_native_fbank + onnxruntime on the same audio).
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { FireRedVAD, StreamingVad } from "../public/js/firered_vad.js";
import { TurnDetector } from "../public/js/turn.js";
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const x = f32(process.argv[2]), ref = f32(process.argv[3]), chunk = +(process.argv[4] || 640);
const dir = new URL("../public/models/vad/", import.meta.url);
ort.env.wasm.numThreads = 1;
const vad = await FireRedVAD.create(ort, readFileSync(new URL("firered_vad.onnx", dir)), JSON.parse(readFileSync(new URL("cmvn.json", dir))));
const sv = new StreamingVad(vad), events = [], all = [];
const td = new TurnDetector({ start: () => events.push(`start @${(td.frame / 100).toFixed(2)}s`), end: (e) => events.push(`end   @${(td.frame / 100).toFixed(2)}s  utterance ${(e.durationMs / 1000).toFixed(2)}s`), discard: () => events.push("discarded") });
for (let i = 0; i < x.length; i += chunk) { const a = x.subarray(i, i + chunk), p = await sv.push(a); all.push(...p); td.push(a, p); }
td.close();
let max = 0; all.forEach((p, i) => (max = Math.max(max, Math.abs(p - ref[i]))));
console.log(`chunk ${chunk} samples: ${all.length} probs (reference ${ref.length}), max |JS - reference| = ${max.toExponential(2)}`);
console.log(events.join("\n"));
if (all.length !== ref.length || max > 2e-2) { console.error("MISMATCH"); process.exit(1); } console.log("OK");
