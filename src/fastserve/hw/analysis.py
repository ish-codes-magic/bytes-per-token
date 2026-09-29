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


def m0_observables(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """The quantities predicted before M0 (benchmarks/predictions/m0.json), computed from one probe run."""
    spec = gpu_spec(records) or {}
    read_bw = measured_bandwidth(records, method="read")
    copy_bw = measured_bandwidth(records, method="copy")
    peaks = measured_peak_flops(records)
    bw_rows = metrics_of(records, "bandwidth")
    in_l2 = [m["gbps_median"] for m in bw_rows if MIB <= m["size_bytes"] <= 32 * MIB]
    tiny = [m["gbps_median"] for m in bw_rows if m["size_bytes"] == 4096]
    m1 = [
        m
        for m in metrics_of(records, "matmul")
        if m["format"] == "bf16" and m["m"] == 1 and m["n"] == 4096 and "tflops_median" in m
    ]
    launch = single(records, "launch_overhead") or {}
    power = single(records, "power") or {}

    def watts(workload: str) -> float | None:
        return (power.get(workload) or {}).get("mean_w")

    bf16_peak_spec = spec.get("peak_flops", {}).get("bf16")
    return {
        "read_bandwidth_gbps": read_bw / 1e9 if read_bw else None,
        "copy_bandwidth_gbps": copy_bw / 1e9 if copy_bw else None,
        "l2_speedup": max(in_l2) * 1e9 / read_bw if in_l2 and read_bw else None,
        "tiny_transfer_gbps": max(tiny) if tiny else None,
        "bf16_peak_tflops": peaks["bf16"] / 1e12 if "bf16" in peaks else None,
        "fp8_over_bf16": peaks["fp8"] / peaks["bf16"] if {"fp8", "bf16"} <= peaks.keys() else None,
        "bf16_m1_pct_of_peak": (
            100 * m1[0]["tflops_median"] * 1e12 / bf16_peak_spec if m1 and bf16_peak_spec else None
        ),
        "bf16_ridge": peaks["bf16"] / read_bw if "bf16" in peaks and read_bw else None,
        "eager_launch_us": launch.get("eager_us_per_kernel"),
        "graph_launch_us": launch.get("graph_us_per_kernel"),
        "idle_power_w": watts("idle"),
        "copy_power_w": watts("copy_1gib"),
        "matmul_power_w": watts("matmul_bf16_8192"),
    }
