"""Rotary position embeddings (RoPE), matching Hugging Face's convention exactly.

Each (x[i], x[i + head_dim/2]) pair of a query or key is rotated by the angle position × θᵢ, with
θᵢ = base^(-2i / head_dim). The dot product of a rotated query and key then depends only on their distance.
"""

from __future__ import annotations

import torch


def inv_freq(head_dim: int, theta: float, device: torch.device | str | None = None) -> torch.Tensor:
    """θᵢ for i = 0 … head_dim/2 − 1, in float32. Shape [head_dim / 2]."""
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    return 1.0 / (theta**exponents)


def cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    """cos/sin tables for the given absolute positions.

    positions: [batch, seq] → cos, sin: [batch, seq, head_dim], computed in float32, then cast to `dtype`
    (as Hugging Face does, so bf16 results match bit for bit).
    """
    theta_i = inv_freq(head_dim, theta, positions.device)  # [head_dim/2]
    freqs = positions[..., None].float() * theta_i  # [batch, seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) → (−x2, x1) on the last dim: the "multiply by i" of a 2-D rotation, for every pair at once."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x: [batch, heads, seq, head_dim] by cos/sin: [batch, seq, head_dim]."""
    cos, sin = cos[:, None], sin[:, None]  # broadcast over heads
    return x * cos + rotate_half(x) * sin
