// Usage: node scripts/test_vad_fbank.mjs <audio.f32> <reference_fbank.f32>
// audio.f32: float32 mono 16 kHz samples in [-1,1]; reference_fbank.f32: kaldi_native_fbank output (frames x 80, float32).
import { readFileSync } from "node:fs";
import { Fbank } from "../public/js/fbank.js";
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const x = f32(process.argv[2]), ref = f32(process.argv[3]);
let worst = 0, n = 0;
for (const chunk of [x.length, 128, 1000, 1]) {            // whole file, worklet-sized blocks, odd sizes, sample by sample
  if (chunk === 1 && x.length > 40000) continue;
  const fb = new Fbank(), frames = [];
  for (let i = 0; i < x.length; i += chunk) frames.push(...fb.push(x.subarray(i, i + chunk)));
  if (frames.length * 80 !== ref.length) { console.error(`chunk ${chunk}: ${frames.length} frames, reference has ${ref.length / 80}`); process.exit(1); }
  let max = 0; frames.forEach((fr, i) => fr.forEach((v, j) => { max = Math.max(max, Math.abs(v - ref[i * 80 + j])); }));
  console.log(`chunk ${String(chunk).padStart(6)}: ${frames.length} frames, max |JS - reference| = ${max.toExponential(2)}`); worst = Math.max(worst, max); n++;
}
if (worst > 5e-3) { console.error("MISMATCH"); process.exit(1); } console.log("OK");
