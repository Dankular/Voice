"""Whisper ASR on ONNX Runtime (no torch/transformers), for transcribing reference audio.

Targets the Optimum/transformers.js export layout used by ``onnx-community/whisper-*``:
``encoder_model*.onnx`` + ``decoder_model_merged*.onnx`` (KV-cache via ``use_cache_branch``).
Greedy decoding, language auto-detected from the first decoder step, 30 s window.
Upstream OmniVoice uses openai/whisper-large-v3-turbo through transformers; any
onnx-community Whisper repo that has these file names works here.
"""
import json
from pathlib import Path

import numpy as np

SR = 16_000
N_FFT, HOP, WINDOW_S = 400, 160, 30
N_SAMPLES = SR * WINDOW_S


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filters(n_mels: int, sr: int = SR, n_fft: int = N_FFT) -> np.ndarray:
    """Slaney-scale, slaney-normalised mel filterbank, shape (n_mels, n_fft//2+1)."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    hz = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    return (w * (2.0 / (hz[2 : n_mels + 2] - hz[:n_mels]))[:, None]).astype(np.float32)


def log_mel(audio: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Whisper log-mel features for one 30 s window, shape (n_mels, 3000)."""
    x = np.zeros(N_SAMPLES, dtype=np.float32)
    x[: min(len(audio), N_SAMPLES)] = audio[:N_SAMPLES]
    x = np.pad(x, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    power = np.abs(np.fft.rfft(x[idx] * win, axis=-1)) ** 2          # (frames, 201)
    mel = filters @ power[:-1].T                                     # drop last frame -> 3000
    spec = np.log10(np.maximum(mel, 1e-10))
    spec = np.maximum(spec, spec.max() - 8.0)
    return ((spec + 4.0) / 4.0).astype(np.float32)


class WhisperONNX:
    def __init__(self, model_dir, encoder="encoder_model_int8.onnx",
                 decoder="decoder_model_merged_int8.onnx", num_threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if num_threads:
            opts.intra_op_num_threads = num_threads
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / encoder), sess_options=opts, providers=prov)
        self.dec = ort.InferenceSession(str(d / decoder), sess_options=opts, providers=prov)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.load(open(d / "config.json"))
        gen = json.load(open(d / "generation_config.json"))
        pre = json.load(open(d / "preprocessor_config.json"))
        self.filters = mel_filters(int(pre.get("feature_size", 80)))
        self.layers = cfg["decoder_layers"]
        self.heads = cfg["decoder_attention_heads"]
        self.head_dim = cfg["d_model"] // self.heads
        self.sot = gen["decoder_start_token_id"]
        self.eos = gen["eos_token_id"]
        self.lang_ids = {int(v): k for k, v in gen["lang_to_id"].items()}
        self.transcribe_id = gen["task_to_id"]["transcribe"]
        self.no_ts = gen["no_timestamps_token_id"]
        self.suppress = np.array(gen.get("suppress_tokens", []), dtype=np.int64)
        self.begin_suppress = np.array(gen.get("begin_suppress_tokens", []), dtype=np.int64)
        self._lang_arr = np.array(sorted(self.lang_ids), dtype=np.int64)

    def _empty_past(self):
        z = np.zeros((1, self.heads, 0, self.head_dim), dtype=np.float32)
        return {f"past_key_values.{i}.{a}.{b}": z for i in range(self.layers)
                for a in ("decoder", "encoder") for b in ("key", "value")}

    def _step(self, ids, enc_h, past, use_cache):
        feed = {"input_ids": np.asarray([ids], dtype=np.int64), "encoder_hidden_states": enc_h,
                "use_cache_branch": np.array([use_cache], dtype=bool), **past}
        out = self.dec.run(None, feed)
        names = [o.name for o in self.dec.get_outputs()]
        logits = out[0][0, -1].astype(np.float32)
        present = {n.replace("present.", "past_key_values."): v for n, v in zip(names[1:], out[1:])}
        return logits, present

    def detect_language(self, audio16k: np.ndarray) -> str:
        enc_h = self.enc.run(None, {"input_features": log_mel(audio16k, self.filters)[None]})[0]
        logits, _ = self._step([self.sot], enc_h, self._empty_past(), False)
        return self.lang_ids[int(self._lang_arr[np.argmax(logits[self._lang_arr])])]

    def transcribe(self, audio16k: np.ndarray, max_new_tokens: int = 224) -> str:
        audio16k = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        enc_h = self.enc.run(None, {"input_features": log_mel(audio16k, self.filters)[None]})[0]

        # language detection from the SOT-only step
        logits, _ = self._step([self.sot], enc_h, self._empty_past(), False)
        lang_tok = int(self._lang_arr[np.argmax(logits[self._lang_arr])])
        prefix = [self.sot, lang_tok, self.transcribe_id, self.no_ts]

        logits, past = self._step(prefix, enc_h, self._empty_past(), False)
        out = []
        for i in range(max_new_tokens):
            logits[self.suppress] = -np.inf
            logits[self.no_ts:] = -np.inf                      # no timestamp tokens
            if i == 0:
                logits[self.begin_suppress] = -np.inf
            tok = int(np.argmax(logits))
            if tok == self.eos:
                break
            out.append(tok)
            logits, new = self._step([tok], enc_h, past, True)
            for k, v in new.items():                           # keep encoder KV from the first pass
                if ".decoder." in k:
                    past[k] = v
        return self.tok.decode(out, skip_special_tokens=True).strip()
