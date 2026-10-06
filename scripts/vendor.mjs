// Copies VAD runtime assets out of node_modules into public/vendor so the
// browser loads everything same-origin (no CDN dependency).
import { cpSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public", "vendor");
mkdirSync(out, { recursive: true });

const vad = join(root, "node_modules/@ricky0123/vad-web/dist");
const ort = join(root, "node_modules/onnxruntime-web/dist");

const files = [
  [vad, "bundle.min.js"],
  [vad, "vad.worklet.bundle.min.js"],
  [vad, "silero_vad_v5.onnx"],
  [vad, "silero_vad_legacy.onnx"],
  [ort, "ort.wasm.min.js"],
  [ort, "ort-wasm-simd-threaded.wasm"],
  [ort, "ort-wasm-simd-threaded.mjs"],
];
for (const [dir, f] of files) {
  const src = join(dir, f);
  if (!existsSync(src)) throw new Error(`missing ${src}`);
  cpSync(src, join(out, f));
}
console.log(`vendored ${files.length} files -> public/vendor`);
