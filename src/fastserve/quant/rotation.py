"""Rotations that spread outliers out without changing what the model computes (QuaRot / SpinQuant style).

**Why it works.** For any orthogonal R (R Rᵀ = I), a linear layer gives the same output if its input is
rotated and its weights are counter-rotated: x W = (x R)(Rᵀ W). A rotated vector keeps its length but mixes
all of its channels, so one huge channel becomes many moderate ones. Grids fit better, and quantization error
drops.

**Where it can be applied inside a transformer.** The residual stream x flows through every layer. Rotate it
once at the embedding (x → x R) and every layer can work in the rotated basis, provided that:
1. every linear that *reads* the stream absorbs R:     W ← W R      (q, k, v, gate, up, the LM head)
2. every linear that *writes* the stream emits in it:  W ← Rᵀ W     (o_proj, down_proj)
3. RMSNorm commutes with R. RMS(x R) = RMS(x) because ‖x R‖ = ‖x‖, but the per-channel weight γ doesn't
   commute, so γ is first folded into the linears that follow each norm, and set to 1.

The LM head can no longer share the embedding matrix: the head absorbs the final norm's γ, and the embedding
doesn't. Real deployments keep both, rotated, which costs the embedding's size once more.

Here nn.Linear stores weights as [out, in] and computes y = x Wᵀ, so "W ← W R" in the math above is
`weight ← weight @ R` (rotating the input columns), and "W ← Rᵀ W" is `weight ← Rᵀ @ weight`.

A **Hadamard matrix** is a convenient R: entries ±1/√n, orthogonal, and it can be applied in O(n log n).
Multiplying by random ±1 signs first ("randomized Hadamard") breaks up any structure that lines up with it.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def hadamard(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The n×n Sylvester Hadamard matrix, normalized to be orthogonal. n must be a power of two."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Sylvester's construction needs a power of two, got {n}")
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < n:  # [[H, H], [H, −H]] doubles the size
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def random_hadamard(n: int, seed: int = 0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """diag(random ±1) · H: still orthogonal, but no longer aligned with any particular input pattern."""
    signs = torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed)).to(dtype) * 2 - 1
    return signs[:, None] * hadamard(n, dtype)


def fold_norm(norm: nn.Module, linears: list[nn.Linear]) -> None:
    """RMSNorm(x)·γ feeding W equals RMSNorm(x) feeding W·diag(γ): move γ into the linears' input columns."""
    gamma = norm.weight.data.double()
    for linear in linears:
        linear.weight.data = (linear.weight.data.double() * gamma).to(linear.weight.dtype)
    norm.weight.data.fill_(1.0)


def untie_lm_head(model: nn.Module) -> None:
    """Give the LM head its own copy of the embedding matrix, so the two can be changed independently."""
    if model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach().clone())


@torch.no_grad()
def fold_all_norms(model: nn.Module) -> None:
    """Fold every RMSNorm's γ into the linears it feeds (QK-norms stay: they act inside each head)."""
    untie_lm_head(model)
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        fold_norm(layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj])
        fold_norm(layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj])
    fold_norm(model.model.norm, [model.lm_head])


@torch.no_grad()
def rotate_residual(model: nn.Module, R: torch.Tensor) -> None:
    """Rotate the residual stream by R [d, d]. The model's outputs don't change (up to float rounding).

    Call fold_all_norms first; this raises if a norm still carries a γ.
    """
    layers = model.model.layers
    norms = [
        model.model.norm,
        *(n for layer in layers for n in (layer.input_layernorm, layer.post_attention_layernorm)),
    ]
    if any(not torch.all(n.weight == 1) for n in norms):
        raise ValueError("fold the RMSNorm weights first (fold_all_norms): γ doesn't commute with a rotation")

    def reads(linear: nn.Linear) -> None:  # input x R: weight [out, d] ← weight @ R
        linear.weight.data = (linear.weight.data.double() @ R).to(linear.weight.dtype)

    def writes(linear: nn.Linear) -> None:  # output rotated: weight [d, in] ← Rᵀ @ weight
        linear.weight.data = (R.T @ linear.weight.data.double()).to(linear.weight.dtype)

    R = R.to(model.lm_head.weight.device, torch.float64)
    emb = model.model.embed_tokens
    emb.weight.data = (emb.weight.data.double() @ R).to(emb.weight.dtype)  # each token's vector, rotated
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj, mlp.gate_proj, mlp.up_proj):
            reads(linear)
        writes(attn.o_proj)
        writes(mlp.down_proj)
    reads(model.lm_head)


def rotate_model(model: nn.Module, seed: int = 0) -> torch.Tensor:
    """Fold the norms and rotate the residual stream by a random Hadamard matrix. Returns R."""
    fold_all_norms(model)
    R = random_hadamard(model.config.hidden_size, seed)
    rotate_residual(model, R)
    return R
