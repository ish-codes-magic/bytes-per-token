"""How fast can this GPU really move bytes? That is the speed limit for decode, and the ceiling of lever 1.

Two probes:
- copy: `dst.copy_(src)` reads N bytes and writes N bytes, so 2N bytes cross the memory bus.
- read: a Triton kernel that reads N bytes and writes almost nothing (one float per block). That's closer to
  what decode does: stream the weights in, emit very little.

Sweeping the transfer size shows three regimes: tiny transfers are latency-bound, mid-size ones are served by
the L2 cache (faster than memory!), and only large ones measure the true memory bandwidth.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

from fastserve.timing import TimingStats, cuda_time_ms, summarize


def _gbps(bytes_moved: float, ms: float) -> float:
    return bytes_moved / (ms / 1e3) / 1e9


def _result(
    method: str, size_bytes: int, bytes_moved: int, stats: TimingStats, **extra: Any
) -> dict[str, Any]:
    return {
        "method": method,
        "size_bytes": size_bytes,
        "bytes_moved": bytes_moved,
        "gbps_median": _gbps(bytes_moved, stats.median_ms),
        "gbps_best": _gbps(bytes_moved, stats.min_ms),
        "timing": stats.to_dict(),
        **extra,
    }


def copy_bandwidth(size_bytes: int, *, iters: int = 50) -> dict[str, Any]:
    """Device-to-device copy of `size_bytes` bytes."""
    import torch

    src = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    stats = summarize(cuda_time_ms(lambda: dst.copy_(src), iters=iters))
    return _result("copy", size_bytes, 2 * size_bytes, stats)


@functools.cache
def _sum_blocks_kernel():
    """Built lazily so importing this module never requires Triton or a GPU."""
    import triton
    import triton.language as tl

    @triton.jit
    def sum_blocks(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)  # one program per BLOCK consecutive floats
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)
        tl.store(out_ptr + pid, tl.sum(x, axis=0))  # one float out per BLOCK floats in

    return sum_blocks


def read_bandwidth(
    size_bytes: int, *, iters: int = 50, blocks: Iterable[int] = (1024, 4096, 8192)
) -> dict[str, Any]:
    """Read-heavy Triton kernel. Tries a few block sizes and keeps the fastest: a tiny autotune."""
    import torch
    import triton

    kernel = _sum_blocks_kernel()
    n = size_bytes // 4  # float32 elements
    x = torch.ones(n, dtype=torch.float32, device="cuda")

    best: tuple[int, TimingStats] | None = None
    for block in blocks:
        grid = (triton.cdiv(n, block),)
        out = torch.empty(grid[0], dtype=torch.float32, device="cuda")
        launch = functools.partial(kernel[grid], x, out, n, BLOCK=block)
        stats = summarize(cuda_time_ms(launch, iters=iters))
        if best is None or stats.median_ms < best[1].median_ms:
            best = (block, stats)

    assert best is not None, "blocks must not be empty"
    block, stats = best
    # Every input float is read once; one float is written per block.
    bytes_moved = 4 * n + 4 * triton.cdiv(n, block)
    return _result("read", size_bytes, bytes_moved, stats, block=block)


def bandwidth_sweep(sizes: Iterable[int], *, iters: int = 50) -> list[dict[str, Any]]:
    """Both probes at every size, smallest to largest."""
    results = []
    for size in sorted(sizes):
        results.append(copy_bandwidth(size, iters=iters))
        results.append(read_bandwidth(size, iters=iters))
    return results
