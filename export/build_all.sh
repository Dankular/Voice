#!/usr/bin/env bash
# Build the cached-prefix, 4-bit OmniVoice LM for the browser. ~10-20 min on CPU, needs ~8 GB RAM and ~8 GB disk.
#   export/build_all.sh [outdir=./kv_out] [hf_repo=Daankular/omnivoice-kv-q4]
# Then host the files in $OUT (Hugging Face repo, any static host with CORS, or public/models/kv for same-origin) and open
# the app with ?kv=<base url>  (or change DEFAULT_KV_BASE in public/js/load_kv.js).
set -euo pipefail
OUT="${1:-kv_out}"; REPO="${2:-Daankular/omnivoice-kv-q4}"; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/ckpt" "$OUT/work"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors huggingface_hub onnx onnx_ir onnxruntime numpy
for f in model.safetensors config.json; do
  [ -f "$OUT/ckpt/$f" ] || curl -L --fail -o "$OUT/ckpt/$f" "https://huggingface.co/k2-fsa/OmniVoice/resolve/main/$f"
done
python "$HERE/export_kv.py" "$OUT/ckpt" "$OUT/work"                                   # fp32 graph + embedding tables
python "$HERE/quantize_kv.py" "$OUT/work/lm_kv_fp32.onnx" "$OUT/work/lm_kv_q4.onnx" 4 32
python "$HERE/pack_tables.py" "$OUT/work" "$OUT/work"
mkdir -p "$OUT/final"
python "$HERE/split_parts.py" "$OUT/work" "$OUT/parts" 80 lm_kv_q4.onnx lm_kv_q4.onnx.data embed_text_int8.bin embed_text_scale.bin embed_audio.bin   # for git/same-origin hosting
cp "$OUT/work"/lm_kv_q4.onnx "$OUT/work"/lm_kv_q4.onnx.data "$OUT/work"/embed_text_int8.bin "$OUT/work"/embed_text_scale.bin "$OUT/work"/embed_audio.bin "$OUT/final/"
echo "Same-origin hosting: copy $OUT/parts/* into public/models/kv/"; echo "Done: $OUT/final  (upload: huggingface-cli upload $REPO $OUT/final .)"
