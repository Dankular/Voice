"""Pick a voice sample -> transcribe it on the fly (Whisper ONNX) -> clone it with OmniVoice for new text."""
import io
import threading
from collections import OrderedDict
from dataclasses import dataclass
from math import gcd
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .asr import WhisperONNX
from .engine import OmniVoiceUnified

MAX_REF_S = 20.0   # upstream warns that references over 20 s hurt quality and speed
END_PUNCT = set(";:,.!?…)]}\"'“”‘’；：，。！？、……）】")


def add_punctuation(text: str) -> str:
    """Upstream's add_punctuation: end the reference transcript with punctuation."""
    text = text.strip()
    if text and text[-1] not in END_PUNCT:
        text += "。" if any("一" <= c <= "鿿" for c in text) else "."
    return text


def decode_audio(data: bytes):
    """Any format libsndfile reads (wav/flac/ogg/mp3...) -> (mono float32, sample_rate)."""
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def _trim_edges(x: np.ndarray, sr: int, lead_ms=100, trail_ms=200, thresh_db=-50.0) -> np.ndarray:
    """Trim leading/trailing silence (numpy stand-in for upstream's pydub remove_silence_edges)."""
    hop = int(sr * 0.01)
    n = len(x) // hop
    if n == 0:
        return x
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-12
    loud = np.where(20 * np.log10(rms) > thresh_db)[0]
    if len(loud) == 0:
        return x
    a = max(0, loud[0] * hop - int(sr * lead_ms / 1000))
    b = min(len(x), (loud[-1] + 1) * hop + int(sr * trail_ms / 1000))
    return x[a:b]


def _cap_length(x: np.ndarray, sr: int, max_s: float) -> np.ndarray:
    """If longer than max_s, cut at the quietest 50 ms frame between 60% and 100% of max_s."""
    if len(x) <= max_s * sr:
        return x
    hop = int(sr * 0.05)
    lo, hi = int(0.6 * max_s * sr) // hop, int(max_s * sr) // hop
    frames = x[: hi * hop].reshape(hi, hop)
    cut = lo + int(np.argmin((frames[lo:hi] ** 2).mean(axis=1)))
    return x[: cut * hop]


@dataclass
class Reference:
    codes: np.ndarray      # (8, Tr)
    rms: float
    text: str              # on-the-fly transcript (with end punctuation)
    seconds: float


class VoiceCloner:
    """Thread-safe; prepared references are cached (LRU) so a voice is transcribed once."""

    def __init__(self, engine: OmniVoiceUnified, asr: WhisperONNX, cache_size: int = 32):
        self.engine, self.asr = engine, asr
        self._cache: "OrderedDict[str, Reference]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()   # one ORT-heavy job at a time

    def prepare(self, audio: bytes, key: Optional[str] = None, ref_text: Optional[str] = None) -> Reference:
        with self._lock:
            if key and key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            x, sr = decode_audio(audio)
            x24 = _trim_edges(_cap_length(resample(x, sr, 24_000), 24_000, MAX_REF_S), 24_000)
            if len(x24) < 24_000 * 1.0:
                raise ValueError("reference audio is too short (<1 s of speech)")
            text = ref_text or self.asr.transcribe(resample(x24, 24_000, 16_000))   # on-the-fly ASR
            if not text.strip():
                raise ValueError("could not transcribe the reference audio")
            codes, rms = self.engine.encode_reference(x24)
            ref = Reference(codes, rms, add_punctuation(text), len(x24) / 24_000)
            if key:
                self._cache[key] = ref
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return ref

    def speak(self, text: str, ref: Reference, language: Optional[str] = None, speed: float = 1.0,
              seed: Optional[int] = None) -> np.ndarray:
        with self._lock:
            return self.engine.synthesize(text, None, language, speed=speed, seed=seed,
                                          ref_tokens=ref.codes, ref_text=ref.text, ref_rms=ref.rms)
