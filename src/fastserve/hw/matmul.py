"""Achieved matrix-multiply throughput per number format: the compute ceiling of the roofline.

For C[M, N] = A[M, K] @ B[K, N] the work is 2·M·N·K FLOPs. We sweep shapes because small matmuls (in decode,
M is the batch size) cannot keep the whole GPU busy, while big ones approach the peak.

Unsupported format/shape combinations are recorded with their error message, never silently skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fastserve.perfmodel.roofline import matmul_flops
from fastserve.timing import cuda_time_ms, summarize

FORMATS = ("bf16", "fp16", "fp8", "int8")


def make_matmul(fmt: str, m: int, n: int, k: int) -> Callable[[], object]:
    """Allocate operands for one format and return a zero-argument function that runs the matmul once."""
    import torch

    dev = "cuda"
    if fmt in ("bf16", "fp16"):
        dtype = torch.bfloat16 if fmt == "bf16" else torch.float16
        a = torch.randn(m, k, device=dev, dtype=dtype)
        b = torch.randn(k, n, device=dev, dtype=dtype)
        c = torch.empty(m, n, device=dev, dtype=dtype)
        return lambda: torch.mm(a, b, out=c)
    if fmt == "fp8":
        a = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn).t()  # cuBLASLt wants B column-major
        one = torch.ones((), device=dev)  # per-tensor scales of 1.0: we time the matmul, not the scaling
        return lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    if fmt == "int8":
        a = torch.randint(-128, 128, (m, k), device=dev, dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device=dev, dtype=torch.int8).t()  # column-major, as above
        return lambda: torch._int_mm(a, b)
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


def matmul_throughput(fmt: str, m: int, n: int, k: int, *, iters: int = 30) -> dict[str, Any]:
    """Time one matmul shape. The result has either `tflops_*` fields or an `error` field."""
    result: dict[str, Any] = {"format": fmt, "m": m, "n": n, "k": k, "flops": matmul_flops(m, n, k)}
    try:
        stats = summarize(cuda_time_ms(make_matmul(fmt, m, n, k), iters=iters))
    except (RuntimeError, TypeError, AttributeError, ValueError) as err:  # format/shape unsupported here
        result["error"] = f"{type(err).__name__}: {str(err).splitlines()[0][:300]}"
        return result
    result["tflops_median"] = result["flops"] / (stats.median_ms / 1e3) / 1e12
    result["tflops_best"] = result["flops"] / (stats.min_ms / 1e3) / 1e12
    result["timing"] = stats.to_dict()
    return result


def matmul_sweep(
    formats: Iterable[str], ms: Iterable[int], nks: Iterable[int], *, iters: int = 30
) -> list[dict[str, Any]]:
    """Every (format, M, N=K) combination. N = K keeps the grid 2-D, which is enough to see the trend."""
    return [matmul_throughput(fmt, m, nk, nk, iters=iters) for fmt in formats for m in ms for nk in nks]
