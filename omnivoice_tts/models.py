"""Fetch the ONNX models and build the voice cloner.

Backbone + codec: a single-graph OmniVoice export whose LM takes a 4D attention mask (bidirectional, as
upstream). Default ct03/omnivoice-onnx-int8hq. NOTE its licence is 'other' (bundles the Higgs Audio 2
community licence) - check it before hosting.
ASR: onnx-community/whisper-base (int8) by default; any onnx-community Whisper repo with the same file names works.
"""
import os
from pathlib import Path

HOME = Path(os.environ.get("OMNIVOICE_HOME", Path.home() / ".cache" / "omnivoice-onnx"))
OMNI_REPO = os.environ.get("OMNIVOICE_REPO", "ct03/omnivoice-onnx-int8hq")
ASR_REPO = os.environ.get("ASR_REPO", "onnx-community/whisper-base")

OMNI_FILES = [
    "omnivoice_lm_int8_hq/model.onnx", "omnivoice_lm_int8_hq/model.onnx_data",
    "audio_tokenizer_decoder_int8/model.onnx", "audio_tokenizer_decoder_int8/model.onnx_data",
    "audio_tokenizer_encoder_int8/model.onnx", "audio_tokenizer_encoder_int8/model.onnx_data",
    "tokenizer.json",
]
ASR_FILES = {"onnx/encoder_model_int8.onnx": "encoder_model_int8.onnx",
             "onnx/decoder_model_merged_int8.onnx": "decoder_model_merged_int8.onnx",
             "tokenizer.json": "tokenizer.json", "config.json": "config.json",
             "generation_config.json": "generation_config.json",
             "preprocessor_config.json": "preprocessor_config.json"}


def load_cloner(num_threads: int = 0):
    """Download (once) and load OmniVoice + Whisper; returns a VoiceCloner."""
    import shutil

    from huggingface_hub import hf_hub_download

    from .asr import WhisperONNX
    from .clone import VoiceCloner
    from .engine import OmniVoiceUnified

    omni = Path(os.environ.get("OMNIVOICE_DIR") or HOME / "omnivoice")
    if not os.environ.get("OMNIVOICE_DIR"):
        for f in OMNI_FILES:
            hf_hub_download(OMNI_REPO, f, local_dir=omni)
    asr = Path(os.environ.get("ASR_DIR") or HOME / "asr")
    if not os.environ.get("ASR_DIR"):
        asr.mkdir(parents=True, exist_ok=True)
        for src, dst in ASR_FILES.items():
            if not (asr / dst).exists():
                shutil.copy(hf_hub_download(ASR_REPO, src), asr / dst)
    engine = OmniVoiceUnified(omni / "omnivoice_lm_int8_hq", omni / "audio_tokenizer_decoder_int8", omni,
                              encoder_dir=omni / "audio_tokenizer_encoder_int8", num_threads=num_threads)
    return VoiceCloner(engine, WhisperONNX(asr, num_threads=num_threads))
