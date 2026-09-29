"""Roofline arithmetic: the two ceilings every kernel lives under (PROJECT.md, Ch 4).

    arithmetic intensity (AI) = FLOPs / bytes moved from memory
    attainable FLOP/s         = min(peak FLOP/s, bandwidth × AI)
    ridge point               = peak FLOP/s / bandwidth

Below the ridge point a kernel is memory-bound; above it, compute-bound.
"""

from __future__ import annotations


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def attainable_flops(intensity: float, *, peak_flops: float, bandwidth: float) -> float:
    """The best FLOP/s a kernel with this arithmetic intensity can reach on this hardware."""
    return min(peak_flops, bandwidth * intensity)


def ridge_point(*, peak_flops: float, bandwidth: float) -> float:
    """The arithmetic intensity where memory-bound turns into compute-bound (FLOPs/byte)."""
    return peak_flops / bandwidth


def matmul_flops(m: int, n: int, k: int) -> int:
    """C[m, n] = A[m, k] @ B[k, n]: every output element needs k multiply-adds, i.e. 2k FLOPs."""
    return 2 * m * n * k


def matmul_bytes(m: int, n: int, k: int, *, a_bytes: float, b_bytes: float, out_bytes: float) -> float:
    """Minimum memory traffic of a matmul: read A and B once, write C once (bytes/element per operand)."""
    return m * k * a_bytes + k * n * b_bytes + m * n * out_bytes


def matmul_intensity(m: int, n: int, k: int, *, a_bytes: float, b_bytes: float, out_bytes: float) -> float:
    """For decode, m is the batch size and the weight (k × n) dominates the bytes, so AI ≈ m."""
    bytes_moved = matmul_bytes(m, n, k, a_bytes=a_bytes, b_bytes=b_bytes, out_bytes=out_bytes)
    return arithmetic_intensity(matmul_flops(m, n, k), bytes_moved)
