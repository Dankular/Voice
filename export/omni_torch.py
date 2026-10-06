"""PyTorch OmniVoice backbone rebuilt from the k2-fsa/OmniVoice checkpoint (no `omnivoice` package needed).
Mirrors OmniVoice._prepare_embed_inputs / forward in k2-fsa/OmniVoice (Apache-2.0)."""
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3Model

NC, VOCAB, MASK = 8, 1025, 1024


class OmniBackbone(nn.Module):
    def __init__(self, ckpt_dir):
        super().__init__()
        cfg = json.load(open(f"{ckpt_dir}/config.json"))
        lc = {k: v for k, v in cfg["llm_config"].items() if k not in ("architectures", "id2label", "label2id")}
        self.llm = Qwen3Model(Qwen3Config(**lc))
        h = self.llm.config.hidden_size
        self.audio_embeddings = nn.Embedding(NC * VOCAB, h)
        self.audio_heads = nn.Linear(h, NC * VOCAB, bias=False)
        self.register_buffer("offsets", torch.arange(NC) * VOCAB)
        sd = load_file(f"{ckpt_dir}/model.safetensors")
        own = {}
        for k, v in sd.items():
            if k.startswith("llm."):
                own[k] = v
            elif k.startswith(("audio_embeddings.", "audio_heads.")):
                own[k] = v
        missing, unexpected = self.load_state_dict(own, strict=False)
        self.missing = [m for m in missing if m != "offsets"]
        self.unexpected = unexpected
        self.eval()

    def embed(self, input_ids, audio_mask):          # [B,8,S], [B,S] -> [B,S,H]
        text = self.llm.embed_tokens(input_ids[:, 0, :])
        shifted = input_ids * audio_mask.unsqueeze(1) + self.offsets.view(1, -1, 1)
        audio = self.audio_embeddings(shifted).sum(dim=1)
        return torch.where(audio_mask.unsqueeze(-1), audio, text)

    def head(self, hidden):                          # [B,S,H] -> [B,8,S,1025]
        b, s, _ = hidden.shape
        return self.audio_heads(hidden).view(b, s, NC, VOCAB).permute(0, 2, 1, 3)

    @torch.no_grad()
    def forward(self, input_ids, audio_mask, attn4d, position_ids):
        out = self.llm(inputs_embeds=self.embed(input_ids, audio_mask), attention_mask=attn4d,
                       position_ids=position_ids)
        return self.head(out.last_hidden_state)
