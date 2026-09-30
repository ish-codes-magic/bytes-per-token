"""Non-uniform 8- and 4-bit formats, simulated in float32: FP8 (E4M3, E5M2) and NF4.

Integer grids space their points evenly. These don't:

- **FP8** keeps a sign, an exponent and a mantissa, like BF16 with fewer bits. Its grid is *logarithmic*:
  every power of two holds the same number of points, so small values get fine steps and large ones coarse
  steps.
  E4M3 (4 exponent bits, 3 mantissa bits) has more precision; E5M2 has more range.
- **NF4** ("NormalFloat", from QLoRA) is a codebook of 16 values placed at quantiles of a normal distribution.
  Weights are roughly bell-shaped, so each level ends up covering about the same number of weights.

Values are scaled into the format's range first (FP8: divide by amax / max_value; NF4: by the block's amax).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import torch


@dataclass(frozen=True)
class FloatFormat:
    """A small floating-point format with an IEEE-style bias and subnormals."""

    name: str
    exp_bits: int
    man_bits: int
    max_value: float  # largest finite value; overflow saturates here

    @property
    def min_normal_exp(self) -> int:
        return 2 - 2 ** (self.exp_bits - 1)  # 1 − bias


# The "fn" variant of E4M3 (used by NVIDIA and PyTorch) has no infinities: its top code is reused, so the
# largest value is 1.75 × 2⁸ = 448 instead of 240.
E4M3 = FloatFormat("fp8_e4m3", exp_bits=4, man_bits=3, max_value=448.0)
E5M2 = FloatFormat("fp8_e5m2", exp_bits=5, man_bits=2, max_value=57344.0)


def round_to_format(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """The nearest value representable in `fmt` (ties to even), saturating at ±max_value. Returns float32.

    Within one binade [2^e, 2^(e+1)) the spacing is 2^(e − man_bits). Below the smallest normal exponent the
    spacing stays at its smallest value: those are the subnormals, which reach down to 0 evenly.
    """
    x = x.float().clamp(-fmt.max_value, fmt.max_value)
    exponent = torch.floor(torch.log2(x.abs().clamp(min=torch.finfo(torch.float32).tiny)))
    step = torch.exp2(exponent.clamp(min=fmt.min_normal_exp) - fmt.man_bits)
    return (torch.round(x / step) * step).clamp(-fmt.max_value, fmt.max_value)  # torch.round: half to even


def representable_values(fmt: FloatFormat) -> torch.Tensor:
    """Every finite value of the format, sorted (zero once). Used to draw the grid over a histogram."""
    bias = 2 ** (fmt.exp_bits - 1) - 1
    positive = []
    for e_field in range(2**fmt.exp_bits):
        for m_field in range(2**fmt.man_bits):
            if e_field == 0:
                value = m_field / 2**fmt.man_bits * 2.0 ** (1 - bias)  # subnormal
            else:
                value = (1 + m_field / 2**fmt.man_bits) * 2.0 ** (e_field - bias)
            if 0 < value <= fmt.max_value:
                positive.append(value)
    positive = sorted(set(positive))
    return torch.tensor([-v for v in reversed(positive)] + [0.0] + positive)


def fp8_fake_quantize(x: torch.Tensor, fmt: FloatFormat = E4M3, granularity: str = "channel") -> torch.Tensor:
    """Scale x so its largest value maps to the format's max, round, and scale back.

    granularity: "tensor" (one scale) or "channel" (one scale per row: an output channel for weights, a token
    for activations).
    """
    x32 = x.float()
    if granularity == "tensor":
        amax = x32.abs().amax()
    elif granularity == "channel":
        amax = x32.abs().amax(dim=-1, keepdim=True)
    else:
        raise ValueError(f"granularity must be 'tensor' or 'channel', got {granularity!r}")
    scale = (amax / fmt.max_value).clamp(min=1e-12)
    return (round_to_format(x32 / scale, fmt) * scale).to(x.dtype)


def nf4_levels(offset: float = 0.9677083) -> list[float]:
    """The 16 NF4 values, built from normal quantiles as in QLoRA (and bitsandbytes' create_normal_map).

    8 positive levels and 7 negative ones at evenly spaced quantiles, plus an exact 0; then everything is
    divided by the largest magnitude so the codebook spans [−1, 1]. `offset` keeps the outermost quantile
    finite.
    Asymmetric on purpose: a 4-bit code has 16 slots, and a symmetric set with a 0 would waste one.
    """
    normal = NormalDist()

    def quantiles(
        n: int,
    ) -> list[float]:  # n − 1 evenly spaced quantiles from `offset` down to (excluding) 0.5
        return [normal.inv_cdf(offset + (0.5 - offset) * i / (n - 1)) for i in range(n - 1)]

    values = quantiles(9) + [0.0] + [-v for v in quantiles(8)]
    biggest = max(abs(v) for v in values)
    return sorted(v / biggest for v in values)


def nf4_fake_quantize(w: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """NF4 with one absmax scale per block of `block_size` consecutive values (bitsandbytes' layout)."""
    levels = torch.tensor(nf4_levels(), device=w.device)  # [16]
    blocks = w.float().reshape(-1, block_size)  # [n_blocks, block]
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    normalized = blocks / absmax  # in [−1, 1]
    nearest = (
        (normalized[..., None] - levels).abs().argmin(dim=-1)
    )  # [n_blocks, block]: index of closest level
    return (levels[nearest] * absmax).reshape(w.shape).to(w.dtype)
