"""Turn raw probe records into the few numbers everything else uses: the *measured* hardware ceilings.

Stdlib only, so docs and reports can be rendered on the laptop without an ML stack.
"""

from __future__ import annotations

from typing import Any

MIB = 2**20


def metrics_of(records: list[dict[str, Any]], experiment: str) -> list[dict[str, Any]]:
    return [r["metrics"] for r in records if r["experiment"] == experiment]


def gpu_spec(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The datasheet entry recorded by the probe (None for GPUs not in fastserve.hw.specs)."""
    info = metrics_of(records, "gpu_info")
    return info[0].get("spec") if info else None


def measured_bandwidth(records: list[dict[str, Any]], *, method: str = "read") -> float | None:
    """Best median bandwidth (bytes/s) over transfers at least 4× the L2 cache, so it's memory, not cache.

    `read` is the default because decode is read-dominated: it streams weights and KV in, and writes little.
    """
    spec = gpu_spec(records)
    min_size = 4 * spec["l2_bytes"] if spec else 256 * MIB
    rows = [
        m for m in metrics_of(records, "bandwidth") if m["method"] == method and m["size_bytes"] >= min_size
    ]
    return max(m["gbps_median"] for m in rows) * 1e9 if rows else None


def measured_peak_flops(records: list[dict[str, Any]]) -> dict[str, float]:
    """Best median FLOP/s per number format over all measured matmul shapes."""
    peaks: dict[str, float] = {}
    for m in metrics_of(records, "matmul"):
        if "tflops_median" in m:
            peaks[m["format"]] = max(peaks.get(m["format"], 0.0), m["tflops_median"] * 1e12)
    return peaks


def single(records: list[dict[str, Any]], experiment: str) -> dict[str, Any] | None:
    """Metrics of an experiment that appears once per run (launch_overhead, power, gpu_info)."""
    rows = metrics_of(records, experiment)
    return rows[0] if rows else None
