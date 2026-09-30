"""W8A8: weights *and* activations in 8 bits (FP8 E4M3 or INT8), so the matmul can use 8-bit tensor cores.

Weights are the easy half: fixed, roughly bell-shaped, one scale per output channel. Activations are the hard
half. A few channels carry values 10–100× larger than the rest (the outlier atlas shows them), and with one
scale per tensor those channels set the step size for everyone.

The options, from cheapest to most work:

    per-tensor static    one scale from calibration, fixed      fastest; outliers set everyone's step
    per-tensor dynamic   one scale per call, from its max       no calibration; the same outlier problem
    per-token dynamic    one scale per token (row)              an outlier *token* only hurts itself, but an
                                                                outlier *channel* is in every token
    SmoothQuant          x Wᵀ = (x / s)(W · s)ᵀ,  s_j = max|x_j|^α / max|W_j|^(1−α)
                         moves part of each channel's range from the activation into the weight

FP8's logarithmic grid tolerates outliers much better than INT8's even grid: large values get coarse steps,
but small values keep fine ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fastserve.quant.equivalence import InputGroup, scale_inputs
from fastserve.quant.formats import E4M3, fp8_fake_quantize, round_to_format
from fastserve.quant.rtn import IntSpec, fake_quantize

Quantizer = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class W8A8Config:
    format: str = "fp8"  # "fp8" (E4M3) | "int8"
    weight_granularity: str = "channel"  # "tensor" | "channel"
    act_granularity: str = "token"  # "tensor" | "token"
    act_static: bool = False  # per-tensor only: use a scale fixed from calibration instead of each call's max

    @property
    def label(self) -> str:
        acts = f"{'static' if self.act_static else 'dynamic'} per-{self.act_granularity}"
        return f"W8A8 {self.format.upper()} (weights per-{self.weight_granularity}, acts {acts})"


def quantize_weight(w: torch.Tensor, cfg: W8A8Config) -> torch.Tensor:
    if cfg.format == "fp8":
        return fp8_fake_quantize(w, E4M3, cfg.weight_granularity)
    return fake_quantize(w, IntSpec(8, cfg.weight_granularity))


def activation_quantizer(cfg: W8A8Config, static_amax: torch.Tensor | None = None) -> Quantizer:
    """A function x [..., in] → fake-quantized x, with the configured granularity and format."""

    def quantize(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]).float()  # [tokens, in]
        if cfg.act_granularity == "token":
            amax = flat.abs().amax(dim=-1, keepdim=True)  # [tokens, 1]
        elif cfg.act_static:
            amax = static_amax.to(flat.device)
        else:
            amax = flat.abs().amax()
        if cfg.format == "fp8":
            scale = (amax / E4M3.max_value).clamp(min=1e-12)
            out = round_to_format(flat / scale, E4M3) * scale  # saturates above a static scale's range
        else:
            scale = (amax / 127).clamp(min=1e-12)
            out = torch.clamp(torch.round(flat / scale), -128, 127) * scale
        return out.reshape(x.shape).to(x.dtype)

    return quantize


class QuantLinear(nn.Module):
    """A linear layer computing with a fake-quantized weight and fake-quantized inputs."""

    def __init__(self, linear: nn.Linear, weight: torch.Tensor, act_quant: Quantizer):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = linear.bias
        self.act_quant = act_quant
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.act_quant(x), self.weight, self.bias)


def smoothing_scale(act_amax: torch.Tensor, weights: list[torch.Tensor], alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant's s_j = max|x_j|^α / max|W_j|^(1−α): α = 0.5 splits each channel's range evenly."""
    w_amax = torch.stack([w.float().abs().amax(dim=0) for w in weights]).amax(
        dim=0
    )  # [in]: over every reader
    s = act_amax.float().clamp(min=1e-5).pow(alpha) / w_amax.clamp(min=1e-5).pow(1 - alpha)
    return s.clamp(min=1e-5)


@torch.no_grad()
def smooth_group(group: InputGroup, act_amax: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Apply SmoothQuant to one input group, in place (function-preserving). Returns s."""
    s = smoothing_scale(act_amax, [lin.weight for lin in group.linears], alpha)
    scale_inputs(group, s)
    return s
