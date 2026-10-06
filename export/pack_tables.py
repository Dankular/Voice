"""Raw little-endian binaries for the browser: int8 text table, f32 per-row scales, f32 audio table."""
import sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
np.load(f"{src}/embed_text_int8.npy").tofile(f"{dst}/embed_text_int8.bin")
np.load(f"{src}/embed_text_scale.npy").astype("<f4").tofile(f"{dst}/embed_text_scale.bin")
np.load(f"{src}/embed_audio_f32.npy").astype("<f4").tofile(f"{dst}/embed_audio.bin")
print("packed")
