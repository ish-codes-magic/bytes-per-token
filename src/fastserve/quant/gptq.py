"""GPTQ: quantize a weight matrix column by column, pushing each column's error onto the columns still ahead.

**Objective.** For one linear layer with calibration inputs X [in, n], find the quantized Ŵ that keeps the
layer's *outputs* close, not the weights:

    minimize ‖W X − Ŵ X‖²      Rows are independent: for each row w, minimize (w − ŵ) H (w − ŵ)ᵀ,
                                with the Hessian H = 2 X Xᵀ [in, in] shared by every row.

**Compensation (Optimal Brain Surgeon).** Quantizing input column q changes the row by e = w_q − ŵ_q. If the
columns F not yet quantized are free to move, the change that best cancels that error in the output is

    δ_F = − e / [H_F⁻¹]_qq · [H_F⁻¹]_q,F           (H_F: H restricted to F, which shrinks as we go)

Inputs that are correlated with input q (large off-diagonal H) absorb the most. With uncorrelated inputs
(H diagonal) there is nothing to compensate, and GPTQ reduces to round-to-nearest.

**GPTQ's three tricks** (Frantar et al., 2022) that make this fast enough for billions of weights:
1. **One column order for all rows**, so every row shares the same sequence of H_F⁻¹.
2. **Cholesky.** The rows [H_F⁻¹]_q,F needed at each step are exactly the rows of the upper Cholesky factor
   of H⁻¹ (scaled), so one factorization replaces a matrix inverse per column.
3. **Lazy batches.** Updates inside a block of 128 columns are applied at once; the rest of W is updated once
   per block. The arithmetic is the same, with far fewer memory round trips.

Plus two options: **dampening** (add a little to H's diagonal so it stays invertible) and **act-order**
(quantize the inputs with the largest H_qq, the most active ones, first while the most freedom remains).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from fastserve.quant.rtn import IntSpec, grouped, round_to_grid, scale_and_zero


class HessianAccumulator:
    """Running H = 2/n · Σ x xᵀ over every calibration token seen by one linear layer's input."""

    def __init__(self, n_features: int, device: torch.device | str = "cpu"):
        self.H = torch.zeros(n_features, n_features, dtype=torch.float32, device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()  # [tokens, in]
        tokens = x.shape[0]
        self.H *= self.n / (self.n + tokens)  # keep it an average as more tokens arrive
        self.n += tokens
        x = math.sqrt(2 / self.n) * x
        self.H += x.T @ x  # [in, in]


@dataclass(frozen=True)
class GPTQConfig:
    spec: IntSpec = field(default_factory=IntSpec)
    block_size: int = 128
    damp: float = 0.01  # fraction of mean(diag H) added to the diagonal
    act_order: bool = False  # quantize the most active input columns first
    static_groups: bool = False  # with act_order: fix each group's grid from the original W, up front


def layer_loss(w: torch.Tensor, w_hat: torch.Tensor, H: torch.Tensor) -> float:
    """The objective ‖(W − Ŵ) X‖² / n expressed with H = 2 X Xᵀ / n: trace(ΔW H ΔWᵀ) / 2."""
    delta = (w - w_hat).float()
    return (delta @ H * delta).sum().item() / 2


def gptq_quantize(
    w: torch.Tensor, H: torch.Tensor, cfg: GPTQConfig, record: list | None = None
) -> torch.Tensor:
    """GPTQ-quantize W [out, in] against the Hessian H [in, in]; returns the dequantized Ŵ (W's dtype).

    `record`, if given, receives one dict per column (in processing order) with the column index, its
    rounding error per row, and a copy of the working matrix after the update (quantized columns so far, the
    rest as updated). Only for small matrices, with block_size ≥ in so no update is deferred.
    """
    spec = cfg.spec
    W = w.detach().float().clone()  # [out, in]
    H = H.detach().float().clone()  # [in, in]
    rows, cols = W.shape
    if spec.granularity == "group" and cfg.block_size % spec.group_size:
        raise ValueError("block_size must be a multiple of group_size, so groups never straddle blocks")

    dead = torch.diag(H) == 0  # inputs that were always 0: nothing to learn, and they'd make H singular
    H[dead, dead] = 1
    W[:, dead] = 0

    # Grids fixed before any update: per-tensor and per-channel always; per-group only with static_groups.
    fixed_grid = None
    if spec.granularity != "group" or cfg.static_groups:
        fixed_grid = scale_and_zero(grouped(W, spec), spec.bits, spec.symmetric, spec.full_range)

    perm = torch.argsort(torch.diag(H), descending=True) if cfg.act_order else torch.arange(cols)
    W, H = W[:, perm], H[perm][:, perm]

    H += cfg.damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    # Upper Cholesky factor of H⁻¹: row i holds what column i's error does to columns i+1… (trick 2)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    Q = torch.zeros_like(W)
    grid = None
    for b1 in range(0, cols, cfg.block_size):  # trick 3: blocks of columns
        b2 = min(b1 + cfg.block_size, cols)
        W1, Hinv1 = W[:, b1:b2].clone(), Hinv[b1:b2, b1:b2]
        Err1 = torch.zeros_like(W1)
        for i in range(b2 - b1):
            col = b1 + i  # position in processing order
            original = perm[col].item()  # the column's index in W as given
            if fixed_grid is not None:
                scale, zero = fixed_grid
                if spec.granularity == "group":  # static groups: the grid of the column's original group
                    g = original // spec.group_size
                    grid = scale[:, g], zero[:, g]
                elif spec.granularity == "channel":
                    grid = scale[:, 0], zero[:, 0]
                else:
                    grid = scale[0, 0].expand(rows, 1), zero[0, 0].expand(rows, 1)
            elif col % spec.group_size == 0:  # a new group starts: its grid from the *updated* weights
                grid = scale_and_zero(
                    W1[:, i : i + spec.group_size], spec.bits, spec.symmetric, spec.full_range
                )
            q = round_to_grid(W1[:, i : i + 1], *grid, spec.bits, spec.symmetric)[:, 0]  # [out]
            Q[:, col] = q
            error = W1[:, i] - q  # [out]: what rounding this column costs each row
            err = error / Hinv1[i, i]
            W1[:, i:] -= err[:, None] @ Hinv1[i : i + 1, i:]  # compensate within the block, right away
            Err1[:, i] = err
            if record is not None:  # with block_size ≥ in, the snapshot shows every update as it happens
                snapshot = torch.cat([Q[:, : col + 1], W1[:, i + 1 :], W[:, b2:]], dim=1)
                record.append(
                    {"column": original, "error": error.clone(), "weights": snapshot[:, torch.argsort(perm)]}
                )
        W[:, b2:] -= Err1 @ Hinv[b1:b2, b2:]  # the rest of W: once per block

    return Q[:, torch.argsort(perm)].to(w.dtype)  # back to the original column order
