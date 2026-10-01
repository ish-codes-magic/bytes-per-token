"""The Qwen3 forward pass in plain PyTorch.

Qwen3 is the Llama architecture plus one addition, QK-norm: a per-head RMSNorm on queries and keys before
RoPE. Parameter names match Hugging Face's exactly (model.layers.0.self_attn.q_proj.weight, ...), so real
checkpoints load with a plain `load_state_dict`.

Shapes use B = batch, Q = new tokens this step, T = tokens in the KV cache, d = hidden size, D = head dim.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastserve.engine import rope
from fastserve.engine.attention import attention, attention_sdpa, causal_mask
from fastserve.engine.config import ModelConfig
from fastserve.engine.kv_cache import KVCache


class RMSNorm(nn.Module):
    """x / sqrt(mean(x²) + eps) · weight, computed in float32 like Hugging Face."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        d, hd, bias = cfg.hidden_size, cfg.head_dim, cfg.attention_bias
        self.q_proj = nn.Linear(d, cfg.num_heads * hd, bias=bias)
        self.k_proj = nn.Linear(d, cfg.num_kv_heads * hd, bias=bias)
        self.v_proj = nn.Linear(d, cfg.num_kv_heads * hd, bias=bias)
        self.o_proj = nn.Linear(cfg.num_heads * hd, d, bias=bias)
        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps)  # QK-norm: normalizes each head's vector
        self.k_norm = RMSNorm(hd, cfg.rms_norm_eps)
        self.kv_policy = None  # M5: what a quantized or evicting cache keeps (fastserve.kv.quant)
        self.attend = attention  # the reference; use_fused_attention() swaps in the fused kernel

    def forward(
        self,
        x: torch.Tensor,  # [B, Q, d]
        cos: torch.Tensor,  # [B, Q, D]
        sin: torch.Tensor,
        positions: torch.Tensor,  # [B, Q]
        cache: KVCache | None,
        meta: Any,
    ) -> torch.Tensor:
        b, q_len, _ = x.shape
        hd = self.cfg.head_dim
        q = self.q_norm(self.q_proj(x).view(b, q_len, self.cfg.num_heads, hd)).transpose(
            1, 2
        )  # [B, Hq, Q, D]
        k = self.k_norm(self.k_proj(x).view(b, q_len, self.cfg.num_kv_heads, hd)).transpose(
            1, 2
        )  # [B, Hkv, Q, D]
        v = self.v_proj(x).view(b, q_len, self.cfg.num_kv_heads, hd).transpose(1, 2)  # [B, Hkv, Q, D]
        q, k = rope.apply(q, cos, sin), rope.apply(k, cos, sin)

        if self.kv_policy is not None:  # round K, V to the cache's format, as it would store them
            k, v = self.kv_policy.store(k, v)
        if cache is not None:  # store this step's keys/values; get back everything so far: [B, Hkv, T, D]
            k, v = cache.update(self.layer_idx, k, v, meta)
        mask = causal_mask(positions, k.shape[2])
        if self.kv_policy is not None:  # an evicting cache no longer holds some slots
            mask = self.kv_policy.visible(mask, positions)
        out = self.attend(q, k, v, mask, scale=hd**-0.5)  # [B, Hq, Q, D]
        return self.o_proj(out.transpose(1, 2).reshape(b, q_len, -1))


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) ⊙ up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then h + mlp(norm(h))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, positions, cache, meta) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, positions, cache, meta)
        return h + self.mlp(self.post_attention_layernorm(h))


class Backbone(nn.Module):
    """Token embeddings → decoder layers → final norm. Named `model` inside CausalLM, as in Hugging Face."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids, positions, cache, meta) -> torch.Tensor:
        x = self.embed_tokens(input_ids)  # [B, Q, d]
        cos, sin = rope.cos_sin(
            positions, self.cfg.head_dim, self.cfg.rope_theta, x.dtype
        )  # shared by all layers
        for layer in self.layers:
            x = layer(x, cos, sin, positions, cache, meta)
        return self.norm(x)


class CausalLM(nn.Module):
    """Backbone + LM head. The LM head shares the embedding matrix when the config ties them."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, Q]
        positions: torch.Tensor | None = None,  # [B, Q] absolute positions; default 0 … Q−1
        cache: KVCache | None = None,
        rows: Any = None,  # which cached sequence each batch row belongs to (see kv_cache.py)
        select: torch.Tensor | None = None,  # [B] index into Q: return logits for that token only
    ) -> torch.Tensor:
        """Logits [B, Q, vocab], or [B, vocab] when `select` is given.

        Without a cache, positions must start at 0 (a plain full forward pass).
        """
        b, q_len = input_ids.shape
        if positions is None:
            positions = torch.arange(q_len, device=input_ids.device).expand(b, q_len)
        meta = None
        if cache is not None:
            rows = torch.arange(b, device=input_ids.device) if rows is None else rows
            meta = cache.prepare(rows, positions)  # computed once, shared by every layer
        hidden = self.model(input_ids, positions, cache, meta)  # [B, Q, d]
        if select is not None:  # e.g. only each prompt's last token: skips a [Q × vocab] matmul
            hidden = hidden[torch.arange(b, device=hidden.device), select]  # [B, d]
        return self.lm_head(hidden)


def use_fused_attention(model: CausalLM, fused: bool = True) -> CausalLM:
    """Run every layer's attention through PyTorch's fused kernel (or back through the reference)."""
    for layer in model.model.layers:
        layer.self_attn.attend = attention_sdpa if fused else attention
    return model
