"""AWQ (activation-aware weight quantization): protect the weights that matter most by scaling them up first.

**Observation** (Lin et al., 2023). A few input channels carry much larger activations than the rest, so the
weights reading them matter most for the output. Keeping ~1% of weights in 16 bits would fix most of the
error, but mixed precision is awkward for hardware.

**Trick.** Scale instead. Multiply input channel j's weights by s_j and divide its activation by s_j (exact:
see equivalence.py). A scaled-up weight is larger relative to its group's step size, so its *relative*
rounding error shrinks. The activation, divided by s_j, carries that error into the output less.

**Search.** s = (mean |x_j|)^α. α = 0 is plain RTN; α → 1 trusts the activation statistics fully. AWQ tries
α on a grid and keeps whichever minimizes the error of the group's actual outputs on calibration data:

    α* = argmin_α ‖ Q(W · s) (x / s)ᵀ − W xᵀ ‖²

**Clipping** (also from the paper). After scaling, each group's range may be shrunk a little: clipping a few
extreme weights costs less than rounding every other weight coarsely. The clip ratio is searched per
(output row, group), again against the output error. As in AutoAWQ, q_proj and k_proj are not clipped: their
outputs feed the attention scores, where errors are amplified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from fastserve.quant.equivalence import InputGroup, scale_inputs
from fastserve.quant.rtn import IntSpec, fake_quantize, grouped, round_to_grid, scale_and_zero


@dataclass(frozen=True)
class AWQConfig:
    spec: IntSpec = field(default_factory=IntSpec)
    grid: int = 20  # α = 0, 1/20, …, 19/20
    clip: bool = True
    clip_grid: int = 20  # shrink ratios 1, 0.95, …
    max_shrink: float = 0.5  # never clip more than half of a group's range


@torch.no_grad()
def search_scale(
    weights: list[torch.Tensor], x: torch.Tensor, cfg: AWQConfig
) -> tuple[torch.Tensor, float, list[float]]:
    """Best per-channel scale for linears sharing input x [tokens, in]: (s [in], α*, loss for every α)."""
    x = x.float()
    act = x.abs().mean(dim=0)  # [in]: how active each input channel is
    reference = torch.cat([x @ w.float().T for w in weights], dim=-1)  # [tokens, Σ out]
    losses, best = [], (float("inf"), None, 0.0)
    for i in range(cfg.grid):
        alpha = i / cfg.grid
        s = act.pow(alpha).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()  # center the scales around 1 (the loss doesn't depend on this)
        out = torch.cat([(x / s) @ fake_quantize(w.float() * s, cfg.spec).T for w in weights], dim=-1)
        loss = (out - reference).pow(2).mean().item()
        losses.append(loss)
        if loss < best[0]:
            best = (loss, s, alpha)
    return best[1], best[2], losses


@torch.no_grad()
def search_clip(w: torch.Tensor, x: torch.Tensor, cfg: AWQConfig) -> torch.Tensor:
    """W with each (row, group) clamped to the range that minimizes that group's output error on x."""
    spec = cfg.spec
    wg = grouped(w.float(), spec)  # [out, n_groups, g]
    xg = x.float().T.reshape(wg.shape[1], wg.shape[2], -1)  # [n_groups, g, tokens]
    reference = torch.einsum("ong,ngt->ont", wg, xg)  # [out, n_groups, tokens]: each group's contribution
    amax = wg.abs().amax(dim=-1, keepdim=True)  # [out, n_groups, 1]
    best_err = torch.full(amax.shape, float("inf"), device=w.device)
    best_max = amax.clone()
    for i in range(int(cfg.max_shrink * cfg.clip_grid)):
        limit = amax * (1 - i / cfg.clip_grid)
        clipped = wg.clamp(-limit, limit)
        q = round_to_grid(
            clipped, *scale_and_zero(clipped, spec.bits, spec.symmetric), spec.bits, spec.symmetric
        )
        err = (torch.einsum("ong,ngt->ont", q, xg) - reference).pow(2).mean(dim=-1, keepdim=True)
        better = err < best_err
        best_err, best_max = torch.where(better, err, best_err), torch.where(better, limit, best_max)
    return wg.clamp(-best_max, best_max).reshape(w.shape).to(w.dtype)


@torch.no_grad()
def awq_group(group: InputGroup, x: torch.Tensor, cfg: AWQConfig) -> tuple[torch.Tensor, dict]:
    """Search and apply the scale for one input group (in place, function-preserving).

    Returns (s, a summary of the search: the chosen α and the loss at every α).
    """
    s, alpha, losses = search_scale([lin.weight for lin in group.linears], x, cfg)
    scale_inputs(group, s.to(x.device))
    return s, {"group": group.name, "alpha": alpha, "losses": losses}


@torch.no_grad()
def awq_quantize_linear(linear: nn.Linear, x_scaled: torch.Tensor | None, cfg: AWQConfig) -> None:
    """Clip (if configured and x is given) and RTN-quantize one linear's (already scaled) weight, in place."""
    w = linear.weight.data
    if cfg.clip and x_scaled is not None:
        w = search_clip(w, x_scaled, cfg)
    linear.weight.data = fake_quantize(w, cfg.spec)
