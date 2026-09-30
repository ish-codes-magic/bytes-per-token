"""Round-to-nearest (RTN) integer quantization: the baseline every smarter method is compared with.

A weight matrix W [out, in] is cut into groups, each group gets its own scale (and, for an asymmetric grid, a
zero-point), and every weight snaps to the nearest point of its group's grid:

    symmetric    s = max|w| / qmax                q = clamp(round(w / s), −qmax − 1, qmax)       ŵ = s·q
    asymmetric   s = (max − min) / (2^b − 1)      q = clamp(round(w / s) + z, 0, 2^b − 1)        ŵ = s·(q − z)
                 z = round(−min / s)              (the integer that stands for a real 0)

with qmax = 2^(b−1) − 1 (7 for INT4). That symmetric grid is the textbook one: 15 levels, and the code −8 goes
unused. Libraries such as compressed-tensors (llm-compressor) use all 16 codes: s = max|w| / 7.5, levels −8…7.
The step is 7% finer, and the largest positive weight is clipped by half a step (`full_range=True`).

Granularity decides how many weights share one scale:

    per-tensor   all of W                         one scale: an outlier anywhere hurts everything
    per-channel  one output row                   the standard for INT8
    per-group    `group_size` consecutive inputs  of one row (e.g. 128): what makes 4 bits work

Rounding error is uniform in [−s/2, s/2], so its mean square is s²/12: every weight in a group pays for the
group's largest value, which is why outliers and coarse granularity hurt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GRANULARITIES = ("tensor", "channel", "group")


@dataclass(frozen=True)
class IntSpec:
    """An integer grid: bit width, how many weights share a scale, and whether the grid is centered on 0."""

    bits: int = 4
    granularity: str = "group"  # "tensor" | "channel" | "group"
    group_size: int = 128
    symmetric: bool = True
    full_range: bool = False  # symmetric only: s = max|w| / (2^b − 1)/2, using every code

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}, got {self.granularity!r}")
        if not 2 <= self.bits <= 8:
            raise ValueError(f"bits must be between 2 and 8, got {self.bits}")

    def weights_per_scale(self, shape: tuple[int, int]) -> int:
        rows, cols = shape
        return {"tensor": rows * cols, "channel": cols, "group": self.group_size}[self.granularity]

    def bits_per_weight(self, shape: tuple[int, int], scale_bits: int = 16) -> float:
        """Storage cost, counting 16-bit scales and, if asymmetric, zero-points stored in `bits` bits."""
        overhead = scale_bits + (0 if self.symmetric else self.bits)
        return self.bits + overhead / self.weights_per_scale(shape)

    @property
    def label(self) -> str:
        size = {"tensor": "per-tensor", "channel": "per-channel", "group": f"g{self.group_size}"}
        kind = ("sym, full range" if self.full_range else "sym") if self.symmetric else "asym"
        return f"INT{self.bits} {size[self.granularity]} {kind}"


def grouped(w: torch.Tensor, spec: IntSpec) -> torch.Tensor:
    """View W [rows, cols] as [n_scales_per_row..., weights sharing a scale]: stats run over the last dim."""
    rows, cols = w.shape
    if spec.granularity == "tensor":
        return w.reshape(1, 1, rows * cols)
    if spec.granularity == "channel":
        return w.reshape(rows, 1, cols)
    if cols % spec.group_size:
        raise ValueError(f"{cols} input features are not divisible by group size {spec.group_size}")
    return w.reshape(rows, cols // spec.group_size, spec.group_size)  # [rows, n_groups, group]


def scale_and_zero(
    x: torch.Tensor, bits: int, symmetric: bool, full_range: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """The grid for the values in x's last dimension: (scale, zero-point), each shaped [..., 1]."""
    x = x.float()
    if symmetric:
        levels = (2**bits - 1) / 2 if full_range else 2 ** (bits - 1) - 1  # 7.5 or 7 for INT4
        scale = x.abs().amax(dim=-1, keepdim=True) / levels
        zero = torch.zeros_like(scale)
    else:
        # Stretch the range to include 0, so a real 0 (e.g. padding, pruned weights) stays exactly 0.
        lo = x.amin(dim=-1, keepdim=True).clamp(max=0)
        hi = x.amax(dim=-1, keepdim=True).clamp(min=0)
        scale = (hi - lo) / (2**bits - 1)
        zero = torch.round(-lo / scale.clamp(min=1e-12))
    return scale.clamp(min=1e-12), zero  # an all-zero group gets a tiny scale instead of a division by 0


def round_to_grid(
    x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, symmetric: bool
) -> torch.Tensor:
    """Snap x to the grid and return the dequantized values (float32)."""
    x = x.float()
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale
    q = torch.clamp(torch.round(x / scale) + zero, 0, 2**bits - 1)
    return (q - zero) * scale


def quantize(w: torch.Tensor, spec: IntSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The integer codes and grid of W: (q [rows, n_scales, weights_per_scale], scale, zero)."""
    x = grouped(w.float(), spec)
    scale, zero = scale_and_zero(x, spec.bits, spec.symmetric, spec.full_range)
    if spec.symmetric:
        qmax = 2 ** (spec.bits - 1) - 1
        return torch.clamp(torch.round(x / scale), -qmax - 1, qmax), scale, zero
    return torch.clamp(torch.round(x / scale) + zero, 0, 2**spec.bits - 1), scale, zero


def fake_quantize(w: torch.Tensor, spec: IntSpec) -> torch.Tensor:
    """RTN-quantize W and dequantize it again: same shape and dtype, values on the grid."""
    x = grouped(w.float(), spec)
    scale, zero = scale_and_zero(x, spec.bits, spec.symmetric, spec.full_range)
    return round_to_grid(x, scale, zero, spec.bits, spec.symmetric).reshape(w.shape).to(w.dtype)
