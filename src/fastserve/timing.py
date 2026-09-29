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


def cuda_time_ms(fn: Callable[[], object], *, warmup: int = 10, iters: int = 50) -> list[float]:
    """Run `fn` `warmup` times untimed, then `iters` times timed with CUDA events.

    Returns one duration in milliseconds per timed iteration. All events are recorded first and the CPU
    synchronizes once at the end, so the timing loop itself adds no stalls between iterations.
    """
    import torch

    for _ in range(warmup):  # compilation, autotuning, allocator and cache warm-up happen here
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()  # wait until the GPU has actually finished everything we queued
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
