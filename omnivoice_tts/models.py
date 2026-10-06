"""Fetch the ONNX files needed for voice design from onnx-community/OmniVoice-Onnx."""
import os
from pathlib import Path

REPO = "onnx-community/OmniVoice-Onnx"
DEFAULT_CACHE = Path(os.environ.get("OMNIVOICE_HOME", Path.home() / ".cache" / "omnivoice-onnx"))

# int4 backbone (~390 MB) + fp16 Higgs decoder (~43 MB). Encoders are only needed for cloning.
BACKBONE = [
    "audio_embeddings_encoder.onnx", "audio_embeddings_encoder.onnx.data",
    "audio_heads_decoder.onnx", "llm_decoder.onnx", "llm_decoder.onnx.data", "tokenizer.json",
]
DECODER = ["higgs_decoder.onnx", "higgs_decoder.onnx.data"]


def ensure_models(cache_dir=None):
    """Download (once) and return (backbone_dir, higgs_dir)."""
    from huggingface_hub import hf_hub_download

    root = Path(cache_dir or DEFAULT_CACHE)
    for f in BACKBONE:
        hf_hub_download(REPO, f"int4/{f}", local_dir=root)
    for f in DECODER:
        hf_hub_download(REPO, f"audio_tokenizer/fp16/{f}", local_dir=root)
    return root / "int4", root / "audio_tokenizer" / "fp16"
