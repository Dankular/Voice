"""Plain-tensor Qwen3 (bidirectional via explicit mask) with explicit key/value I/O, for ONNX export.
Weights come from the HF Qwen3Model inside OmniBackbone. Two entry points:
  full(embeds, mask, pos)              -> hidden, [(k, v) per layer]      (whole sequence, returns all K/V)
  step(embeds, mask, pos, past_kv)     -> hidden                           (only new positions; attends past + self)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms(x, w, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(v + eps))


def rot(x, cos, sin):          # x [B,H,S,D]; HF rotate_half convention
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class QwenKV(nn.Module):
    def __init__(self, llm):
        super().__init__()
        c = llm.config
        self.layers = llm.layers
        self.norm = llm.norm
        self.nh, self.nkv, self.hd, self.eps = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        theta = c.rope_parameters["rope_theta"] if getattr(c, "rope_parameters", None) else c.rope_theta
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd)), persistent=False)

    def rope(self, pos):       # pos [B,S] int64 -> cos,sin [B,1,S,D]
        f = pos[..., None].float() * self.inv_freq            # [B,S,D/2]
        emb = torch.cat((f, f), dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]

    def _layers(self, x, mask, pos, past):
        cos, sin = self.rope(pos)
        B, S, _ = x.shape
        outs = []
        for i, L in enumerate(self.layers):
            a = L.self_attn
            h = rms(x, L.input_layernorm.weight, self.eps)
            q = a.q_norm(a.q_proj(h).view(B, S, self.nh, self.hd)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(B, S, self.nkv, self.hd)).transpose(1, 2)
            v = a.v_proj(h).view(B, S, self.nkv, self.hd).transpose(1, 2)
            q, k = rot(q, cos, sin), rot(k, cos, sin)
            outs.append((k, v))
            if past is not None:
                k = torch.cat([past[i][0], k], dim=2); v = torch.cat([past[i][1], v], dim=2)
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
            att = att.masked_fill(~mask, torch.finfo(att.dtype).min).softmax(-1)
            x = x + a.o_proj((att @ v).transpose(1, 2).reshape(B, S, self.nh * self.hd))
            h = rms(x, L.post_attention_layernorm.weight, self.eps)
            x = x + L.mlp.down_proj(F.silu(L.mlp.gate_proj(h)) * L.mlp.up_proj(h))
        return rms(x, self.norm.weight, self.eps), outs

    def full(self, embeds, mask, pos):
        return self._layers(embeds, mask, pos, None)

    def step(self, embeds, mask, pos, past):
        return self._layers(embeds, mask, pos, past)[0]
