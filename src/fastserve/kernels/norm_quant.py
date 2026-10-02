"""Kernel 1: RMSNorm fused with per-token INT8 activation quantization.

**What it computes.** For every token (row) x of a [rows, d] hidden-state tensor:

    y = x / sqrt(mean(x²) + eps) · weight      scale = max|y| / 127      codes = round(y / scale)  (int8)

the same result as `reference.rms_norm_int8`, and what vLLM's W8A8-INT8 path computes with two kernels:
`rms_norm` and then `scaled_int8_quant` (vLLM 0.30 fuses RMSNorm only with FP8 quantization).

**What it fuses.** Three passes over the row (sum of squares; max of the normalized row; round) that would
each be a trip to memory. Here the row is loaded once into registers and all three run on that copy.

**Tiling.** One program per token. A program holds its whole row (d ≤ BLOCK elements, BLOCK the next power of
two), so no value is read twice and rows never interact.

**Bytes moved per token** (d elements, 16-bit input):

    unfused   norm: read 2d, write 2d      quantize: read 2d, write d + 4      = 7d + 4
    fused     read 2d, write d + 4                                             = 3d + 4      (2.3× fewer)

The weight vector (2d bytes) is shared by every row and stays in cache.

**Expected bound.** About 12 FLOPs per 3 bytes: far left of the ridge, so memory-bound once the rows no longer
fit in the L2 cache. For a few rows (decode) the kernel is launch-bound instead: the gain there is one launch
instead of two, and it only shows inside a CUDA graph, where Triton's Python launcher is out of the way.
"""

from __future__ import annotations

import functools

import torch


@functools.cache
def _kernel():
    """Built lazily so importing this module never requires Triton or a GPU."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def rms_norm_int8_kernel(
        x_ptr,  # [rows, d] float16 / bfloat16 / float32
        weight_ptr,  # [d]
        codes_ptr,  # [rows, d] int8, written
        scale_ptr,  # [rows] float32, written
        x_row_stride,  # elements between consecutive rows of x
        d,
        eps,
        BLOCK: tl.constexpr,  # power of two ≥ d
    ):
        row = tl.program_id(0).to(tl.int64)  # one program per token
        cols = tl.arange(0, BLOCK)
        inside = cols < d  # BLOCK may overshoot d: masked elements load as 0 and are never stored

        # The only read of this row: everything below runs on the copy in registers.
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=inside, other=0.0).to(tl.float32)

        y = x * tl.rsqrt(tl.sum(x * x, axis=0) / d + eps) * weight  # pass 1: RMSNorm
        scale = tl.maximum(tl.max(tl.abs(y), axis=0) / 127.0, 1e-12)  # pass 2: this token's grid
        codes = libdevice.rint(y / scale)  # pass 3: round half to even, like torch.round
        codes = tl.minimum(tl.maximum(codes, -128.0), 127.0).to(tl.int8)

        tl.store(codes_ptr + row * d + cols, codes, mask=inside)
        tl.store(scale_ptr + row, scale)

    return rms_norm_int8_kernel


def rms_norm_int8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token INT8 in one kernel: (codes int8 [..., d], scale float32 [..., 1]).

    x: [..., d] on a CUDA device. `out` reuses (codes [rows, d], scale [rows]) buffers from an earlier call,
    which keeps allocations out of timed and graph-captured regions.
    """
    import triton

    d = x.shape[-1]
    rows = x.reshape(-1, d)
    if not rows.is_contiguous():
        rows = rows.contiguous()
    n = rows.shape[0]
    if out is None:
        out = (
            torch.empty((n, d), dtype=torch.int8, device=x.device),
            torch.empty(n, dtype=torch.float32, device=x.device),
        )
    codes, scale = out
    _kernel()[(n,)](
        rows,
        weight,
        codes,
        scale,
        rows.stride(0),
        d,
        eps,
        BLOCK=triton.next_power_of_2(d),
        num_warps=num_warps,
    )
    return codes.view(x.shape), scale.view(*x.shape[:-1], 1)
