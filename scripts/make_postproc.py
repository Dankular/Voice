"""Build the tiny GPU post-processing graphs used between LM steps (needs `pip install onnx onnxruntime numpy`).

post_cfg.onnx  : cond[1,8,S,1025], uncond[1,8,T,1025], t0:int64[1], g:float[]  ->  pred[1,8,T] int64, conf[1,8,T] float
post_nocfg.onnx: cond[1,8,S,1025], t0:int64[1]                                  ->  pred, conf
Only ops that ORT-Web's WebGPU EP supports (LogSoftmax is not on that list, so log-softmax is spelled out).
Equivalent to upstream's _predict_tokens_with_scoring: log-softmax both branches, CFG mix, log-softmax again,
then argmax / max over the 1024 real codes (mask id 1024 excluded).
Running this script also checks the graphs against a numpy reference.
"""
import numpy as np
import onnx
from onnx import TensorProto as TP, helper as h, numpy_helper as nh

OPSET = 18


def build(cfg: bool, path: str):
    nodes, inits = [], []
    c = lambda name, arr: (inits.append(nh.from_array(np.asarray(arr), name)), name)[1]

    def lsm(x, tag):
        a = c(f"ax_{tag}", np.array([-1], dtype=np.int64))
        nodes.extend([
            h.make_node("ReduceMax", [x, a], [f"{tag}_m"], keepdims=1),
            h.make_node("Sub", [x, f"{tag}_m"], [f"{tag}_d"]),
            h.make_node("Exp", [f"{tag}_d"], [f"{tag}_e"]),
            h.make_node("ReduceSum", [f"{tag}_e", a], [f"{tag}_s"], keepdims=1),
            h.make_node("Log", [f"{tag}_s"], [f"{tag}_l"]),
            h.make_node("Sub", [f"{tag}_d", f"{tag}_l"], [f"{tag}_o"]),
        ])
        return f"{tag}_o"

    inputs = [h.make_tensor_value_info("cond", TP.FLOAT, [1, 8, "S", 1025])]
    if cfg:
        inputs.append(h.make_tensor_value_info("uncond", TP.FLOAT, [1, 8, "T", 1025]))
    inputs.append(h.make_tensor_value_info("t0", TP.INT64, [1]))
    if cfg:
        inputs.append(h.make_tensor_value_info("g", TP.FLOAT, []))

    # target region of the conditional logits: [t0, t0+T) along the sequence axis; T = a fixed-size read of
    # `uncond` for cfg, otherwise passed implicitly by slicing to the end of the cond tensor's own length is not
    # possible, so the no-cfg graph takes T from t0-independent `tlen`.
    if cfg:
        nodes += [h.make_node("Shape", ["uncond"], ["ushape"]),
                  h.make_node("Gather", ["ushape", c("i2", np.array([2], dtype=np.int64))], ["tlen"])]
    else:
        inputs.append(h.make_tensor_value_info("tlen", TP.INT64, [1]))
    nodes += [h.make_node("Add", ["t0", "tlen"], ["t1"]),
              h.make_node("Slice", ["cond", "t0", "t1", c("ax2", np.array([2], dtype=np.int64))], ["cl_raw"])]
    cl = lsm("cl_raw", "c")
    if cfg:
        ul = lsm("uncond", "u")
        nodes += [h.make_node("Sub", [cl, ul], ["diff"]), h.make_node("Mul", ["diff", "g"], ["gd"]),
                  h.make_node("Add", [cl, "gd"], ["mixin"])]
        mixed = lsm("mixin", "m")
    else:
        mixed = cl
    nodes += [h.make_node("Slice", [mixed, c("s0", np.array([0], dtype=np.int64)), c("e1024", np.array([1024], dtype=np.int64)),
                                    c("ax3", np.array([3], dtype=np.int64))], ["top"]),
              h.make_node("ArgMax", ["top"], ["pred"], axis=-1, keepdims=0),
              h.make_node("ReduceMax", ["top", c("axl", np.array([-1], dtype=np.int64))], ["conf"], keepdims=0)]
    g = h.make_graph(nodes, "post_cfg" if cfg else "post_nocfg", inputs,
                     [h.make_tensor_value_info("pred", TP.INT64, [1, 8, "T"]),
                      h.make_tensor_value_info("conf", TP.FLOAT, [1, 8, "T"])], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def lsm_np(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def check():
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    S, T, t0, g = 70, 23, 41, 2.0
    cond = (rng.standard_normal((1, 8, S, 1025)) * 3).astype(np.float32)
    unc = (rng.standard_normal((1, 8, T, 1025)) * 3).astype(np.float32)
    cl, ul = lsm_np(cond[:, :, t0:t0 + T]), lsm_np(unc)
    for cfg in (True, False):
        mixed = lsm_np(cl + g * (cl - ul)) if cfg else cl
        top = mixed[..., :1024]
        s = ort.InferenceSession("public/models/" + ("post_cfg.onnx" if cfg else "post_nocfg.onnx"), providers=["CPUExecutionProvider"])
        feed = {"cond": cond, "t0": np.array([t0], np.int64)}
        feed.update({"uncond": unc, "g": np.array(g, np.float32)} if cfg else {"tlen": np.array([T], np.int64)})
        pred, conf = s.run(None, feed)
        assert (pred == top.argmax(-1)).all(), "argmax mismatch"
        print(("cfg  " if cfg else "nocfg"), "pred equal; max |conf diff| =", float(np.abs(conf - top.max(-1)).max()))


if __name__ == "__main__":
    build(True, "public/models/post_cfg.onnx")
    build(False, "public/models/post_nocfg.onnx")
    check()
