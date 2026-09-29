"""Reference attention: causal masking by absolute position, and grouped-query attention (GQA).

Deliberately written out in plain PyTorch (no fused kernels) so every step is visible. Faster versions come
later and are tested against this one.
"""

from __future__ import annotations

import torch


def causal_mask(positions: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Key slot j is visible to a query at absolute position p iff j <= p.

    positions: [batch, q_len] → mask: [batch, 1, q_len, kv_len] (bool, broadcast over heads).
    Masking by *position* (not by index within the batch) lets sequences of different lengths share one batch:
    slots a sequence hasn't written yet sit beyond its position and are hidden automatically.
    """
    slots = torch.arange(kv_len, device=positions.device)
    return (slots[None, None, :] <= positions[:, :, None])[:, None]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[batch, kv_heads, seq, head_dim] → [batch, kv_heads × n_rep, seq, head_dim].

    KV head h serves query heads h·n_rep … h·n_rep + n_rep − 1 (the same grouping as Hugging Face).
    """
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    return x[:, :, None].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float
) -> torch.Tensor:
    """softmax(q·kᵀ·scale + mask)·v.

    q: [batch, q_heads, q_len, head_dim]   k, v: [batch, kv_heads, kv_len, head_dim]
    mask: [batch, 1, q_len, kv_len]        → [batch, q_heads, q_len, head_dim]
    """
    group = q.shape[1] // k.shape[1]
    k, v = repeat_kv(k, group), repeat_kv(v, group)
    scores = (q @ k.transpose(-2, -1)) * scale  # [batch, q_heads, q_len, kv_len]
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)  # softmax in fp32 for stability
    return probs @ v
