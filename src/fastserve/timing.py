"""Timing helpers for GPU code.

GPU work is asynchronous: a Python call only *queues* a kernel and returns immediately. A wall clock around
that call measures the queueing, not the work. `cuda_time_ms` times on the GPU's own timeline with CUDA
events instead, and `summarize` turns repeated measurements into the statistics we report.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TimingStats:
    """Summary of repeated measurements, in milliseconds."""

    n: int
    median_ms: float
    mean_ms: float
    min_ms: float
    p90_ms: float
    p99_ms: float
    stdev_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated percentile, q in [0, 100] (the same convention as numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(times_ms: Sequence[float]) -> TimingStats:
    """Median, spread and tail of repeated timings. We report the median, never a single run."""
    if not times_ms:
        raise ValueError("no timings to summarize")
    return TimingStats(
        n=len(times_ms),
        median_ms=statistics.median(times_ms),
        mean_ms=statistics.fmean(times_ms),
        min_ms=min(times_ms),
        p90_ms=percentile(times_ms, 90),
        p99_ms=percentile(times_ms, 99),
        stdev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    )


def cuda_time_ms(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    iters: int = 50,
    flush_l2_bytes: int = 0,
    flush_by: str = "write",
) -> list[float]:
    """Run `fn` `warmup` times untimed, then `iters` times timed with CUDA events.

    Returns one duration in milliseconds per timed iteration. All events are recorded first and the CPU
    synchronizes once at the end, so the timing loop itself adds no stalls between iterations.

    `flush_l2_bytes > 0` overwrites a scratch buffer of that size before every timed iteration, outside the
    timed region, which evicts fn's data from the L2 cache. Use it for "cold" measurements: in real decode the
    weights are far bigger than L2 and always come from memory, so warm-cache timings would flatter them.

    `flush_by="read"` evicts by *reading* the scratch buffer instead. A write leaves the cache full of
    modified lines, and fn then also pays for writing them back to memory as it evicts them (M7 measured
    100–150 µs on the L4). In a real decode step the cache was last filled by reads of other layers, so
    reading is the closer imitation. "write" stays the default because M0–M6 were measured with it.
    """
    if flush_by not in ("write", "read"):
        raise ValueError(f"flush_by must be 'write' or 'read', got {flush_by!r}")
    import torch

    scratch = torch.empty(flush_l2_bytes, dtype=torch.uint8, device="cuda") if flush_l2_bytes else None
    for _ in range(warmup):  # compilation, autotuning and allocator warm-up happen here
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        if scratch is not None:  # queued before `start`, so the flush itself is not timed
            scratch.zero_() if flush_by == "write" else scratch.sum()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()  # wait until the GPU has actually finished everything we queued
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def cuda_graph_time_ms(
    fn: Callable[[], object], *, calls: int = 20, warmup: int = 5, iters: int = 30
) -> list[float]:
    """Time `fn` the way a serving engine runs its decode step: captured in a CUDA graph and replayed.

    A replay re-issues the recorded kernel launches with one driver call, so Python, the framework's
    dispatch and (for Triton) the launcher's argument handling all drop out. `calls` copies of `fn` are
    captured so the replay's own fixed cost is spread thin. Returns milliseconds per call of `fn`.
    `fn` must not synchronize with the GPU (no `.item()`), which capture does not allow.
    """
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # PyTorch requires a warm-up run on a side stream before capture
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    return [ms / calls for ms in cuda_time_ms(graph.replay, warmup=warmup, iters=iters)]
