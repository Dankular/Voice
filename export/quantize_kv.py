"""fp32 KV graph -> MatMulNBits weight-only quantized graph (WebGPU has tuned kernels for MatMulNBits).
usage: quantize_kv.py in.onnx out.onnx [bits=4] [block=32]. The audio head (1024x8200) stays fp32."""
import sys
import onnx
from onnx import numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig

src, dst = sys.argv[1], sys.argv[2]
bits = int(sys.argv[3]) if len(sys.argv) > 3 else 4
block = int(sys.argv[4]) if len(sys.argv) > 4 else 32
model = onnx.load(src)
shapes = {t.name: tuple(t.dims) for t in model.graph.initializer}
skip = [n.name for n in model.graph.node if n.op_type == "MatMul" and shapes.get(n.input[1]) in ((1024, 8200), (8200, 1024))]
print("excluding audio head nodes:", skip)
cfg = RTNWeightOnlyQuantConfig(bits=bits, is_symmetric=True, accuracy_level=4) if False else None
quant = MatMulNBitsQuantizer(model, bits=bits, block_size=block, is_symmetric=True, accuracy_level=4, nodes_to_exclude=skip)
quant.process()
quant.model.save_model_to_file(dst, True)
print("saved", dst)
