"""Hand-build a low-dispatch version of the KV graph, reusing the existing 4-bit weights (no new big files).

usage: build_fused.py lm_kv_q4.onnx(+.data next to it) out_dir     -> out_dir/lm_fused.onnx (references lm_kv_q4.onnx.data)

Why: the traced graph has ~3250 kernel launches per forward; ORT-Web's WebGPU backend is CPU-dispatch bound for small
inputs, so a cached step cost ~100 ms regardless of token count. This graph has ~800 launches:
  SimplifiedLayerNormalization / SkipSimplifiedLayerNormalization (RMSNorm + residual fused), RotaryEmbedding (contrib),
  MatMulNBits (existing weights), attention with grouped heads (q viewed as [8 kv heads, 2*Q rows]; no K/V repeat),
  no Shape/Cast/Unsqueeze ops, 1/sqrt(d) folded into the q_norm weight.
I/O contract (differs from the traced graph):
  in : inputs_embeds[1,Q,1024] f32, attention_bias[1,1,2Q,L] f32 (0 / -1e9; the Q x L bias duplicated for the 2 query groups),
       position_ids[1,Q] i64, past_key_values.{i}.key [1,8,128,Sp] (K TRANSPOSED), past_key_values.{i}.value [1,8,Sp,128]; L = Sp+Q
  out: logits[1,8,Q,1025], present.{i}.key [1,8,128,Q], present.{i}.value [1,8,Q,128]
"""
import sys, os, math
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as h, numpy_helper as nh
from onnx.external_data_helper import load_external_data_for_tensor

src, out_dir = sys.argv[1], sys.argv[2]
NL, NH, NKV, D, HID, EPS, MAXPOS, THETA = 28, 16, 8, 128, 1024, 1e-6, 4096, 1e6
old = onnx.load(src, load_external_data=False)
oinit = {t.name: t for t in old.graph.initializer}
onode = {n.name: n for n in old.graph.node}
base = os.path.dirname(os.path.abspath(src))

inits, nodes = {}, []
def add_init(name, arr=None, tensor=None):
    if tensor is not None:
        t = onnx.TensorProto(); t.CopyFrom(tensor); t.name = name; inits[name] = t       # keeps external-data reference
    else:
        inits[name] = nh.from_array(np.asarray(arr), name)
    return name
def small(name):                                                                          # load a (small) old tensor's values
    t = onnx.TensorProto(); t.CopyFrom(oinit[name])
    if t.external_data: load_external_data_for_tensor(t, base)
    return nh.to_array(t)
def const(name, arr): return add_init(name, np.asarray(arr))
def node(op, ins, outs, domain="", **kw): nodes.append(h.make_node(op, ins, outs, domain=domain, **kw)); return outs

# --- constants ---
inv = 1.0 / (THETA ** (np.arange(0, D, 2, dtype=np.float64) / D))
ang = np.arange(MAXPOS, dtype=np.float64)[:, None] * inv[None, :]
const("cos_cache", np.cos(ang).astype(np.float32)); const("sin_cache", np.sin(ang).astype(np.float32))
const("shp_q", np.array([0, 0, NH, D], np.int64)); const("shp_kv", np.array([0, 0, NKV, D], np.int64))
const("shp_qg", np.array([1, NKV, -1, D], np.int64)); const("shp_o4", np.array([1, NH, -1, D], np.int64))
const("shp_o3", np.array([0, 0, NH * D], np.int64)); const("shp_logits", np.array([0, 0, 8, 1025], np.int64))

def mm4(layer, name, x, out):
    suffix = "" if layer == 0 else f"_{layer}"
    n = onode[f"/{name}{suffix}/MatMul_Q4"]
    for i in (1, 2):
        add_init(n.input[i], tensor=oinit[n.input[i]])
    a = {x.name: (x.i if x.type == onnx.AttributeProto.INT else None) for x in n.attribute}
    node("MatMulNBits", [x, n.input[1], n.input[2]], [out], domain="com.microsoft", K=a["K"], N=a["N"], bits=4,
         block_size=a["block_size"], accuracy_level=4)
    return out

def norm_w(layer, kind):
    nm = {"in": "input_layernorm", "post": "post_attention_layernorm"}[kind]
    k = f"q.layers.{layer}.{nm}.weight"; add_init(f"w_{kind}_{layer}", tensor=oinit[k]); return f"w_{kind}_{layer}"

# --- graph ---
x_res, normed = None, None
for L in range(NL):
    p = f"l{L}"
    if L == 0:
        node("SimplifiedLayerNormalization", ["inputs_embeds", norm_w(0, "in")], [f"{p}_n1"], epsilon=EPS, axis=-1, stash_type=1)
        res = "inputs_embeds"
    else:
        res = x_res                                                    # set at the end of the previous layer
    n1 = f"{p}_n1" if L == 0 else normed
    q = mm4(L, "q_proj", n1, f"{p}_q"); k = mm4(L, "k_proj", n1, f"{p}_k"); v = mm4(L, "v_proj", n1, f"{p}_v")
    qn_w = (small(f"q.layers.{L}.self_attn.q_norm.weight") / math.sqrt(D)).astype(np.float32)   # fold the attention scale
    add_init(f"{p}_qnw", qn_w); add_init(f"{p}_knw", small(f"q.layers.{L}.self_attn.k_norm.weight"))
    node("Reshape", [q, "shp_q"], [f"{p}_q4"]); node("SimplifiedLayerNormalization", [f"{p}_q4", f"{p}_qnw"], [f"{p}_qn"], epsilon=EPS, axis=-1, stash_type=1)
    node("Transpose", [f"{p}_qn"], [f"{p}_qt"], perm=[0, 2, 1, 3])
    node("RotaryEmbedding", [f"{p}_qt", "position_ids", "cos_cache", "sin_cache"], [f"{p}_qr"], domain="com.microsoft", interleaved=0, is_packed_batching=0, num_heads=0, rotary_embedding_dim=0)
    node("Reshape", [k, "shp_kv"], [f"{p}_k4"]); node("SimplifiedLayerNormalization", [f"{p}_k4", f"{p}_knw"], [f"{p}_kn"], epsilon=EPS, axis=-1, stash_type=1)
    node("Transpose", [f"{p}_kn"], [f"{p}_kt"], perm=[0, 2, 1, 3])
    node("RotaryEmbedding", [f"{p}_kt", "position_ids", "cos_cache", "sin_cache"], [f"{p}_kr"], domain="com.microsoft", interleaved=0, is_packed_batching=0, num_heads=0, rotary_embedding_dim=0)
    node("Transpose", [f"{p}_kr"], [f"present.{L}.key"], perm=[0, 1, 3, 2])             # K stored transposed [1,8,D,Q]
    node("Reshape", [v, "shp_kv"], [f"{p}_v4"]); node("Transpose", [f"{p}_v4"], [f"present.{L}.value"], perm=[0, 2, 1, 3])   # [1,8,Q,D]
    node("Concat", [f"past_key_values.{L}.key", f"present.{L}.key"], [f"{p}_kc"], axis=3)
    node("Concat", [f"past_key_values.{L}.value", f"present.{L}.value"], [f"{p}_vc"], axis=2)
    node("Reshape", [f"{p}_qr", "shp_qg"], [f"{p}_qg"])                                 # [1,8,2Q,D]
    node("MatMul", [f"{p}_qg", f"{p}_kc"], [f"{p}_s"])                                   # [1,8,2Q,L]
    node("Add", [f"{p}_s", "attention_bias"], [f"{p}_sb"]); node("Softmax", [f"{p}_sb"], [f"{p}_pr"], axis=-1)
    node("MatMul", [f"{p}_pr", f"{p}_vc"], [f"{p}_av"])                                  # [1,8,2Q,D]
    node("Reshape", [f"{p}_av", "shp_o4"], [f"{p}_ao"]); node("Transpose", [f"{p}_ao"], [f"{p}_at"], perm=[0, 2, 1, 3])
    node("Reshape", [f"{p}_at", "shp_o3"], [f"{p}_a3"])
    o = mm4(L, "o_proj", f"{p}_a3", f"{p}_o")
    node("SkipSimplifiedLayerNormalization", [o, res, norm_w(L, "post")], [f"{p}_n2", "", "", f"{p}_r2"], domain="com.microsoft", epsilon=EPS)
    g = mm4(L, "gate_proj", f"{p}_n2", f"{p}_g"); u = mm4(L, "up_proj", f"{p}_n2", f"{p}_u")
    node("Sigmoid", [g], [f"{p}_sg"]); node("Mul", [g, f"{p}_sg"], [f"{p}_si"]); node("Mul", [f"{p}_si", u], [f"{p}_m"])
    d = mm4(L, "down_proj", f"{p}_m", f"{p}_d")
    nxt_gamma = norm_w(L + 1, "in") if L + 1 < NL else None
    if nxt_gamma is None:
        add_init("w_final", tensor=oinit["q.norm.weight"]); nxt_gamma = "w_final"
    node("SkipSimplifiedLayerNormalization", [d, f"{p}_r2", nxt_gamma], [f"l{L + 1}_n1" if L + 1 < NL else "final_n", "", "", f"l{L}_res"], domain="com.microsoft", epsilon=EPS)
    normed, x_res = (f"l{L + 1}_n1" if L + 1 < NL else "final_n"), f"l{L}_res"
    if L + 1 < NL: normed = f"l{L + 1}_n1"

head = onode["/audio_heads/MatMul"]; add_init(head.input[1], tensor=oinit[head.input[1]])
node("MatMul", ["final_n", head.input[1]], ["logits_flat"]); node("Reshape", ["logits_flat", "shp_logits"], ["logits_4"])
node("Transpose", ["logits_4"], ["logits"], perm=[0, 2, 1, 3])

f32 = TP.FLOAT
vi = lambda n, t, s: h.make_tensor_value_info(n, t, s)
inputs = [vi("inputs_embeds", f32, [1, "Q", HID]), vi("attention_bias", f32, [1, 1, "Q2", "L"]), vi("position_ids", TP.INT64, [1, "Q"])]
outputs = [vi("logits", f32, [1, 8, "Q", 1025])]
for i in range(NL):
    inputs += [vi(f"past_key_values.{i}.key", f32, [1, NKV, D, "Sp"]), vi(f"past_key_values.{i}.value", f32, [1, NKV, "Sp", D])]
    outputs += [vi(f"present.{i}.key", f32, [1, NKV, D, "Q"]), vi(f"present.{i}.value", f32, [1, NKV, "Q", D])]
g = h.make_graph(nodes, "omnivoice_fused", inputs, outputs, list(inits.values()))
m = h.make_model(g, opset_imports=[h.make_opsetid("", 18), h.make_opsetid("com.microsoft", 1)])
m.ir_version = 9
os.makedirs(out_dir, exist_ok=True)
onnx.save(m, os.path.join(out_dir, "lm_fused.onnx"))
print("nodes:", len(nodes), "| kernel-launching (excl. Reshape):", sum(1 for n in nodes if n.op_type != "Reshape"))
