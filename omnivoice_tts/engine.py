"""CPU/GPU ONNX Runtime engine for OmniVoice *voice design* (no reference audio).

This follows the generation procedure of upstream k2-fsa/OmniVoice
(``OmniVoice._prepare_inference_inputs`` / ``_generate_iterative``, Apache-2.0)
on top of the sub-models in onnx-community/OmniVoice-Onnx:

  audio_embeddings_encoder -> llm_decoder -> audio_heads_decoder  (x num_step)
  higgs_decoder (codes -> 24 kHz waveform)

Differences from the ONNX repo's own ``inference.py`` (which only does a
simplified greedy loop): the prompt carries the ``<|lang_*|>`` / ``<|instruct_*|>``
style tokens, text is wrapped in ``<|text_start|>/<|text_end|>``, decoding uses
classifier-free guidance with the upstream time-step schedule, and the target
length comes from upstream's duration estimator.

Not implemented: voice cloning (needs the Higgs encoders) and upstream's
pydub-based silence removal / chunked long-text generation.
"""
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .duration import RuleDurationEstimator

SAMPLE_RATE = 24_000
NUM_CODEBOOKS = 8
MASK_ID = 1024
HOP = 960
_ORT2NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16}

_NONVERBAL = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _combine_text(text: str, ref_text: Optional[str] = None) -> str:
    t = (ref_text.strip() + " " + text.strip()) if ref_text else text.strip()
    t = re.sub(r"[\r\n]+", "", t)
    t = t.replace("（", "(").replace("）", ")")
    t = re.sub(r"[ \t]+", " ", t)
    cjk = r"[一-鿿]"
    return re.sub(rf"(?<={cjk})\s+|\s+(?={cjk})", "", t)


class OmniVoiceONNX:
    def __init__(self, model_dir, higgs_dir, providers=("CPUExecutionProvider",), num_threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir, higgs_dir = Path(model_dir), Path(higgs_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if num_threads:
            opts.intra_op_num_threads = num_threads

        def sess(p):
            if not p.exists():
                raise FileNotFoundError(p)
            return ort.InferenceSession(str(p), sess_options=opts, providers=list(providers))

        self.embed = sess(model_dir / "audio_embeddings_encoder.onnx")
        self.llm = sess(model_dir / "llm_decoder.onnx")
        self.heads = sess(model_dir / "audio_heads_decoder.onnx")
        self.decoder = sess(higgs_dir / "higgs_decoder.onnx")
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.estimator = RuleDurationEstimator()

        llm_in = self.llm.get_inputs()
        self._llm_names = {i.name for i in llm_in}
        self._llm_dtype = _ORT2NP.get(next(i.type for i in llm_in if i.name == "inputs_embeds"), np.float32)
        self._past = [i.name for i in llm_in if "past" in i.name]
        self._heads_dtype = _ORT2NP.get(
            next(i.type for i in self.heads.get_inputs() if i.name == "hidden_states"), np.float32
        )

    # ---- tokenisation -------------------------------------------------
    def _ids(self, text: str) -> list:
        return self.tok.encode(text, add_special_tokens=False).ids

    def _text_ids(self, text: str) -> list:
        """Tokenise, encoding non-verbal tags (e.g. [laughter]) standalone, like upstream."""
        out, last = [], 0
        for m in _NONVERBAL.finditer(text):
            if m.start() > last:
                out += self._ids(text[last:m.start()])
            out += self._ids(m.group())
            last = m.end()
        if last < len(text):
            out += self._ids(text[last:])
        return out

    # ---- one backbone forward ----------------------------------------
    def _forward(self, input_ids: np.ndarray, audio_mask: np.ndarray) -> np.ndarray:
        emb = self.embed.run(["inputs_embeds"], {"input_ids": input_ids, "audio_mask": audio_mask})[0]
        B, S, _ = emb.shape
        feed = {"inputs_embeds": emb.astype(self._llm_dtype, copy=False)}
        if "attention_mask" in self._llm_names:
            feed["attention_mask"] = np.ones((B, S), dtype=np.int64)
        if "position_ids" in self._llm_names:
            feed["position_ids"] = np.arange(S, dtype=np.int64)[None, :]
        for n in self._past:
            feed[n] = np.zeros((B, 8, 0, 128), dtype=self._llm_dtype)
        hid = self.llm.run(["hidden_states"], feed)[0]
        logits = self.heads.run(["logits"], {"hidden_states": hid.astype(self._heads_dtype, copy=False)})[0]
        return logits.astype(np.float32, copy=False)  # (B, 8, S, 1025)

    # ---- public API ---------------------------------------------------
    def synthesize(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: Optional[str] = None,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        layer_penalty_factor: float = 5.0,
        position_temperature: float = 5.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
        ref_tokens: Optional[np.ndarray] = None,
        ref_text: Optional[str] = None,
        ref_rms: Optional[float] = None,
    ) -> np.ndarray:
        """Return a float32 mono waveform at 24 kHz.

        Voice cloning: pass ``ref_tokens`` (8, Tr) from :meth:`encode_reference` and its transcript ``ref_text``.
        """
        if not text or not text.strip():
            raise ValueError("text is empty")
        rng = np.random.default_rng(seed)

        # target length (upstream falls back to this reference when there is no ref audio)
        Tr = 0 if ref_tokens is None else int(ref_tokens.shape[-1])
        if Tr and ref_text:
            T = max(1, int(self.estimator.estimate_duration(text, ref_text, Tr)))
        else:  # upstream's fallback when there is no usable reference
            T = max(1, int(self.estimator.estimate_duration(text, "Nice to meet you.", 25)))
        if speed > 0 and speed != 1.0:
            T = max(1, int(T / speed))

        style = ("<|denoise|>" if Tr else "") + (
            f"<|lang_start|>{language or 'None'}<|lang_end|>"
            f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
        )
        prefix = self._ids(style) + self._text_ids(
            f"<|text_start|>{_combine_text(text, ref_text if Tr else None)}<|text_end|>")
        c_len = len(prefix) + Tr + T
        t0 = len(prefix) + Tr          # start of the target region

        cond = np.full((1, NUM_CODEBOOKS, c_len), MASK_ID, dtype=np.int64)
        cond[0, :, : len(prefix)] = np.asarray(prefix, dtype=np.int64)[None, :]
        cond_mask = np.zeros((1, c_len), dtype=bool)
        if Tr:
            cond[0, :, len(prefix):t0] = ref_tokens
        cond_mask[0, len(prefix):] = True   # reference + target positions are audio (upstream)
        unc = np.full((1, NUM_CODEBOOKS, T), MASK_ID, dtype=np.int64)
        unc_mask = np.ones((1, T), dtype=bool)

        # unmasking schedule (upstream _get_time_steps + per-step counts)
        ts = np.linspace(0.0, 1.0, num_step + 1)
        ts = t_shift * ts / (1 + (t_shift - 1) * ts)
        total, rem, sched = T * NUM_CODEBOOKS, T * NUM_CODEBOOKS, []
        for s in range(num_step):
            k = rem if s == num_step - 1 else min(math.ceil(total * (ts[s + 1] - ts[s])), rem)
            sched.append(int(k))
            rem -= int(k)

        tokens = np.full((NUM_CODEBOOKS, T), MASK_ID, dtype=np.int64)
        layer_ids = np.arange(NUM_CODEBOOKS, dtype=np.float32)[:, None]

        for s in range(num_step):
            k = sched[s]
            if k <= 0:
                continue
            c_logits = self._cond_logits(cond, cond_mask, t0, s)  # (8,T,1025)
            if guidance_scale != 0:
                u_logits = self._forward(unc, unc_mask)[0, :, :T, :]
                cl, ul = _log_softmax(c_logits), _log_softmax(u_logits)
                logp = _log_softmax(cl + guidance_scale * (cl - ul))
            else:
                logp = _log_softmax(c_logits)
            logp[..., MASK_ID] = -np.inf
            pred = logp.argmax(-1)                      # (8,T)  class_temperature=0 (greedy)
            scores = logp.max(-1) - layer_ids * layer_penalty_factor
            if position_temperature > 0:
                u = rng.random(scores.shape, dtype=np.float32)
                scores = scores / position_temperature - np.log(-np.log(u + 1e-10) + 1e-10)
            scores = np.where(tokens != MASK_ID, -np.inf, scores)
            top = np.argpartition(-scores.ravel(), k - 1)[:k]
            flat = tokens.ravel()
            flat[top] = pred.ravel()[top]
            tokens = flat.reshape(NUM_CODEBOOKS, T)
            cond[0, :, t0:] = tokens
            unc[0] = tokens

        return _post_process(self._decode(tokens), ref_rms=ref_rms)

    def _cond_logits(self, cond, cond_mask, t0, step):
        """Logits at the target positions of the conditional pass (hook: subclasses may reuse cached prefix state)."""
        return self._forward(cond, cond_mask)[0, :, t0:, :]

    def _decode(self, tokens: np.ndarray) -> np.ndarray:
        """(8, T) codes -> float32 waveform at 24 kHz."""
        return self.decoder.run(["waveform_24k"], {"codes": tokens[:, None, :]})[0].reshape(-1).astype(np.float32)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _post_process(wav: np.ndarray, ref_rms: Optional[float] = None, pad_s: float = 0.1,
                  fade_s: float = 0.1) -> np.ndarray:
    """Upstream post-processing minus silence removal: scale to the reference loudness if it was quiet,
    peak-normalise to 0.5 when there is no reference, then fade and pad."""
    if ref_rms is not None:
        if ref_rms < 0.1:
            wav = wav * ref_rms / 0.1
    else:
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        if peak > 1e-6:
            wav = wav / peak * 0.5
    k = min(int(fade_s * SAMPLE_RATE), wav.size // 2)
    if k > 0:
        wav = wav.copy()
        wav[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        wav[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    pad = np.zeros(int(pad_s * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([pad, wav, pad])


class OmniVoiceUnified(OmniVoiceONNX):
    """Backend for single-graph exports whose LM takes ``attention_mask[B,1,S,S]`` and ``position_ids``
    (e.g. ct03/omnivoice-onnx-int8hq: omnivoice_lm_*/model.onnx + audio_tokenizer_decoder_*/model.onnx).

    Unlike the genai-built backbone in onnx-community/OmniVoice-Onnx (causal: verified by perturbation test),
    this LM attends bidirectionally when given a full-ones mask, as upstream does.
    """

    def __init__(self, lm_dir, decoder_dir, tokenizer_dir, encoder_dir=None, providers=("CPUExecutionProvider",),
                 num_threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if num_threads:
            opts.intra_op_num_threads = num_threads

        def sess(d):
            return ort.InferenceSession(str(Path(d) / "model.onnx"), sess_options=opts, providers=list(providers))

        self.lm = sess(lm_dir)
        self.decoder = sess(decoder_dir)
        self.encoder = sess(encoder_dir) if encoder_dir else None
        self.tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
        self.estimator = RuleDurationEstimator()

    def _forward(self, input_ids, audio_mask):
        S = input_ids.shape[-1]
        out = self.lm.run(["logits"], {
            "input_ids": input_ids, "audio_mask": audio_mask,
            "attention_mask": np.ones((input_ids.shape[0], 1, S, S), dtype=bool),
            "position_ids": np.broadcast_to(np.arange(S, dtype=np.int64), (input_ids.shape[0], S)).copy(),
        })[0]
        return out.astype(np.float32, copy=False)

    def _decode(self, tokens):
        return self.decoder.run(["audio"], {"audio_codes": tokens[None]})[0].reshape(-1).astype(np.float32)

    def encode_reference(self, wav24k: np.ndarray):
        """Reference waveform (24 kHz mono float32) -> (codes (8, Tr), rms). Mirrors upstream's prompt prep
        except silence removal / long-clip trimming (callers should pass <= ~20 s)."""
        if self.encoder is None:
            raise RuntimeError("no encoder_dir given; voice cloning unavailable")
        wav = np.asarray(wav24k, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        if 0 < rms < 0.1:
            wav = wav * 0.1 / rms
        # codec hop is 960 samples (25 fps; measured). Lengths that aren't a multiple make the encoder fail
        # (acoustic/semantic frame mismatch), so clip like upstream does.
        wav = wav[: len(wav) // HOP * HOP]
        codes = self.encoder.run(["audio_codes"], {"audio": wav[None, None, :]})[0][0]
        return codes.astype(np.int64), rms


class OmniVoiceKV(OmniVoiceUnified):
    """Same decoding as OmniVoiceUnified, but the LM is the single KV-exposing graph from export/export_kv.py
    (fp32 or MatMulNBits-quantized). The prefix (style + text + reference) K/V is computed once in a full pass at
    step 0; later steps run only the target positions against it (target K/V columns from the full pass are masked).
    Embedding lookups happen here (numpy) from the tables written by the exporter.
    """

    def __init__(self, lm_path, tables_dir, decoder_dir, tokenizer_dir, encoder_dir=None, num_threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if num_threads:
            opts.intra_op_num_threads = num_threads
        d = Path(tables_dir)
        self.lm = ort.InferenceSession(str(lm_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(Path(decoder_dir) / "model.onnx"), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.encoder = (ort.InferenceSession(str(Path(encoder_dir) / "model.onnx"), sess_options=opts,
                                             providers=["CPUExecutionProvider"]) if encoder_dir else None)
        self.tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
        self.estimator = RuleDurationEstimator()
        self.txt_i8 = np.load(d / "embed_text_int8.npy")
        self.txt_sc = np.load(d / "embed_text_scale.npy")
        self.aud = np.load(d / "embed_audio_f32.npy")
        self._past = None
        self._out_names = [o.name for o in self.lm.get_outputs()]
        self.n_full = self.n_step = 0

    def _embed(self, ids, audio_mask):                       # ids (1,8,S) int64, audio_mask (1,S) bool -> (1,S,1024)
        ids, am = ids[0], audio_mask[0]
        out = self.txt_i8[ids[0]].astype(np.float32) * self.txt_sc[ids[0]][:, None]
        off = (np.arange(NUM_CODEBOOKS) * 1025)[:, None]
        aud = self.aud[ids * am[None, :] + off].sum(axis=0)   # non-audio ids zeroed first, as upstream (S, 1024)
        return np.where(am[:, None], aud, out)[None].astype(np.float32)

    def _run(self, embeds, mask, pos, past):
        empty = np.zeros((1, 8, 0, 128), np.float32)
        feed = {"inputs_embeds": embeds, "attention_mask": mask, "position_ids": pos}
        for i in range(28):
            for kv in ("key", "value"):
                feed[f"past_key_values.{i}.{kv}"] = past[f"{i}.{kv}"] if past else empty
        return self.lm.run(None, feed)

    def _forward(self, ids, audio_mask):                     # no-cache pass (unconditional branch)
        S = ids.shape[-1]
        return self._run(self._embed(ids, audio_mask), np.ones((1, 1, S, S), bool),
                         np.arange(S, dtype=np.int64)[None], None)[0]

    def _cond_logits(self, cond, cond_mask, t0, step):
        S = cond.shape[-1]
        if step == 0 or self._past is None:                   # full pass: also caches K/V of every position
            self.n_full += 1
            out = self._run(self._embed(cond, cond_mask), np.ones((1, 1, S, S), bool),
                            np.arange(S, dtype=np.int64)[None], None)
            self._past = {n.replace("present.", ""): v for n, v in zip(self._out_names[1:], out[1:])}
            self._emb = None
            return out[0][0, :, t0:, :]
        self.n_step += 1
        Tq = S - t0
        mask = np.zeros((1, 1, Tq, S + Tq), bool)
        mask[..., :t0] = True                                 # cached prefix columns
        mask[..., S:] = True                                  # the new target positions
        out = self._run(self._embed(cond[:, :, t0:], cond_mask[:, t0:]), mask,
                        np.arange(t0, S, dtype=np.int64)[None], self._past)
        return out[0][0]
