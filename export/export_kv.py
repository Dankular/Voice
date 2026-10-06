"""Export the OmniVoice backbone as ONE ONNX graph with explicit KV I/O.

  inputs : inputs_embeds[1,Q,1024] f32, attention_mask[1,1,Q,Sp+Q] bool, position_ids[1,Q] i64,
           past_key_values.{i}.key / .value [1,8,Sp,128] f32 (Sp may be 0)           i = 0..27
  outputs: logits[1,8,Q,1025], present.{i}.key / .value [1,8,Q,128] (K/V of the Q new positions)

Full pass = Sp 0 (all positions); step pass = Sp = full length, Q = target positions only.
Embedding lookups are done outside the graph (JS) so the 620 MB text table is not an initializer.
Also writes the embedding tables for the JS side.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from omni_torch import OmniBackbone, NC, VOCAB
from qwen_kv import QwenKV

CK, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
m = OmniBackbone(CK)
q = QwenKV(m.llm)
NL = len(q.layers)


class G(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.q, self.m = q, m

    def forward(self, embeds, mask, pos, *past):
        pk = [(past[2 * i], past[2 * i + 1]) for i in range(NL)] if past[0].shape[2] > 0 else None
        h, kv = self.q._layers(embeds, mask, pos, pk)
        out = [self.m.head(h)]
        for k, v in kv:
            out += [k, v]
        return tuple(out)


names_in = ["inputs_embeds", "attention_mask", "position_ids"] + [f"past_key_values.{i}.{kv}" for i in range(NL) for kv in ("key", "value")]
names_out = ["logits"] + [f"present.{i}.{kv}" for i in range(NL) for kv in ("key", "value")]
Q, Sp = 40, 24
args = (torch.randn(1, Q, 1024), torch.ones(1, 1, Q, Sp + Q, dtype=torch.bool), torch.arange(Sp, Sp + Q)[None],
        *[torch.randn(1, 8, Sp, 128) for _ in range(2 * NL)])
dyn = {"inputs_embeds": {1: "Q"}, "attention_mask": {2: "Q", 3: "KV"}, "position_ids": {1: "Q"}, "logits": {2: "Q"}}
for i in range(NL):
    for kv in ("key", "value"):
        dyn[f"past_key_values.{i}.{kv}"] = {2: "Sp"}; dyn[f"present.{i}.{kv}"] = {2: "Q"}
with torch.no_grad():
    torch.onnx.export(G().eval(), args, f"{OUT}/lm_kv_fp32.onnx", input_names=names_in, output_names=names_out,
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
print("exported graph")

# ---- embedding tables for the JS side ----
txt = m.llm.embed_tokens.weight.detach().float().numpy()                  # [151676, 1024]
scale = np.abs(txt).max(1) / 127.0 + 1e-12
np.save(f"{OUT}/embed_text_int8.npy", np.round(txt / scale[:, None]).astype(np.int8))
np.save(f"{OUT}/embed_text_scale.npy", scale.astype(np.float32))
np.save(f"{OUT}/embed_audio_f32.npy", m.audio_embeddings.weight.detach().float().numpy())   # [8200, 1024]
np.save(f"{OUT}/audio_head_f32.npy", m.audio_heads.weight.detach().float().numpy())
json.dump({"layers": NL, "kv_heads": 8, "head_dim": 128, "hidden": 1024}, open(f"{OUT}/meta.json", "w"))
print("tables written")
