"""AWQ (activation-aware weight quantization): protect the weights that matter most by scaling them up first.

**Observation** (Lin et al., 2023). A few input channels carry much larger activations than the rest, so the
weights reading them matter most for the output. Keeping ~1% of weights in 16 bits would fix most of the
error, but mixed precision is awkward for hardware.

**Trick.** Scale instead. Multiply input channel j's weights by s_j and divide its activation by s_j (exact:
see equivalence.py). A scaled-up weight is larger relative to its group's step size, so its *relative*
rounding error shrinks. The activation, divided by s_j, carries that error into the output less.

**Search.** The scale is built from the mean activation magnitude x̄_j of each channel:

    s = x̄^α                     the paper's form: α = 0 is plain RTN, α → 1 trusts the activations fully
    s = x̄^α / w̄^(1−α)           "duo scaling" (AutoAWQ's and llm-compressor's default): also uses w̄, each
                                 channel's mean weight magnitude (relative to its group's max)

α goes over a grid, and AWQ keeps the α that best preserves the output of the *parent module*, the block the
group's linears belong to (for q/k/v, the whole attention block, including its softmax):

    α* = argmin_α ‖ parent with weights Q(W · s) / s  −  parent with weights W ‖²   on calibration inputs

**Clipping** (optional, from the paper and AutoAWQ; llm-compressor has none). After scaling, each group's
range may be shrunk a little: clipping a few extreme weights costs less than rounding every other weight
coarsely.
The clip ratio is searched per (output row, group) against that group's output error. q_proj and k_proj are
not clipped: their outputs feed the attention scores, where errors are amplified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from fastserve.quant.equivalence import InputGroup, scale_inputs
from fastserve.quant.rtn import IntSpec, fake_quantize, grouped, round_to_grid, scale_and_zero


@dataclass(frozen=True)
class AWQConfig:
    spec: IntSpec = field(default_factory=IntSpec)
    grid: int = 20  # α = 0, 1/20, …, 19/20
    duo_scaling: bool = True
    clip: bool = False
    clip_grid: int = 20  # shrink ratios 1, 0.95, …
    max_shrink: float = 0.5  # never clip more than half of a group's range


def weight_means(weights: list[torch.Tensor], spec: IntSpec) -> torch.Tensor:
    """w̄ [in]: each input channel's mean |weight|, measured relative to its group's largest |weight|."""
    w = torch.cat([x.float() for x in weights])  # [Σ out, in]: every layer reading this input
    g = grouped(w, spec).abs()
    return (g / (g.amax(dim=-1, keepdim=True) + 1e-6)).reshape(w.shape).mean(dim=0)


@torch.no_grad()
def search_scale(
    linears: list[nn.Linear], x_mean: torch.Tensor, run_parent: Callable[[], torch.Tensor], cfg: AWQConfig
) -> tuple[torch.Tensor, float, list[float]]:
    """The best per-channel scale for linears sharing one input: (s [in], α*, loss at every α).

    `run_parent()` runs the parent module on stored calibration inputs and returns its output; during the
    search each linear temporarily computes with Q(W · s) / s, which equals Q(W · s) applied to x / s.
    """
    originals = [lin.weight.data.clone() for lin in linears]
    reference = run_parent().float()
    w_mean = weight_means(originals, cfg.spec) if cfg.duo_scaling else None
    x_mean = x_mean.float()
    losses, best = [], (float("inf"), None, 0.0)
    try:
        for i in range(cfg.grid):
            alpha = i / cfg.grid
            s = x_mean.pow(alpha)
            if w_mean is not None:
                s = s / (w_mean.pow(1 - alpha) + 1e-4)
            s = s.clamp(min=1e-4)
            s = s / (s.max() * s.min()).sqrt()  # center the scales around 1 (the loss doesn't depend on this)
            for lin, w in zip(linears, originals, strict=True):
                lin.weight.data = (fake_quantize(w.float() * s, cfg.spec) / s).to(w.dtype)
            loss = (run_parent().float() - reference).pow(2).mean().item()
            losses.append(loss)
            if loss < best[0]:
                best = (loss, s.clone(), alpha)
    finally:
        for lin, w in zip(linears, originals, strict=True):
            lin.weight.data = w
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
        grid = scale_and_zero(clipped, spec.bits, spec.symmetric, spec.full_range)
        q = round_to_grid(clipped, *grid, spec.bits, spec.symmetric)
        err = (torch.einsum("ong,ngt->ont", q, xg) - reference).pow(2).mean(dim=-1, keepdim=True)
        better = err < best_err
        best_err, best_max = torch.where(better, err, best_err), torch.where(better, limit, best_max)
    return wg.clamp(-best_max, best_max).reshape(w.shape).to(w.dtype)


@torch.no_grad()
def awq_group(
    group: InputGroup, x_mean: torch.Tensor, run_parent: Callable[[], torch.Tensor], cfg: AWQConfig
) -> tuple[torch.Tensor, dict]:
    """Search and apply the scale for one input group (in place, function-preserving).

    Returns (s, a summary of the search: the chosen α and the loss at every α).
    """
    s, alpha, losses = search_scale(group.linears, x_mean, run_parent, cfg)
    scale_inputs(group, s)
    return s, {"group": group.name, "alpha": alpha, "losses": losses}


@torch.no_grad()
def awq_quantize_linear(linear: nn.Linear, x_scaled: torch.Tensor | None, cfg: AWQConfig) -> None:
    """Clip (if configured and x is given) and RTN-quantize one linear's (already scaled) weight, in place."""
    w = linear.weight.data
    if cfg.clip and x_scaled is not None:
        w = search_clip(w, x_scaled, cfg)
    linear.weight.data = fake_quantize(w, cfg.spec)
