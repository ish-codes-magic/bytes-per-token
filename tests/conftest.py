"""Shared test setup: GPU tests skip themselves on CPU machines, and a fake probe run for CPU tests."""

from __future__ import annotations

from typing import Any

import pytest

from fastserve.hw.specs import SPECS
from fastserve.results import make_record


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if has_gpu:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def probe_records() -> list[dict[str, Any]]:
    """A small hand-made probe run shaped like a real one on an L4, for testing analysis, tables, figures."""
    env = {"gpu": "NVIDIA L4", "driver": "d", "cuda_runtime": "c", "torch": "t", "triton": "tr", "cpu": "x"}
    git = {"commit": "0123456789abcdef", "dirty": False}

    def rec(experiment: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return make_record(experiment, metrics, run_id="testrun", env=env, git=git)

    idle_telemetry = {"sm_clock_mhz": 210, "max_sm_clock_mhz": 2040, "temperature_c": 58}
    records = [rec("gpu_info", {"spec": SPECS["L4"].to_dict(), "telemetry": idle_telemetry})]
    for log2 in (12, 20, 24, 28, 30):  # 4 KiB ... 1 GiB; only 2^28 and 2^30 exceed 4 × the 48 MiB L2
        for method, base in (("copy", 200.0), ("read", 250.0)):
            gbps = base + log2
            records.append(
                rec(
                    "bandwidth",
                    {
                        "method": method,
                        "size_bytes": 2**log2,
                        "bytes_moved": 2**log2,
                        "gbps_median": gbps,
                        "gbps_best": gbps + 1,
                    },
                )
            )
    for fmt, peak in (("bf16", 90.0), ("fp8", 170.0), ("int8", 150.0)):
        for m in (1, 8192):
            for nk in (1024, 8192):
                metrics: dict[str, Any] = {"format": fmt, "m": m, "n": nk, "k": nk, "flops": 2 * m * nk * nk}
                if fmt == "int8" and m == 1:
                    metrics["error"] = "RuntimeError: m must be > 16"
                else:
                    tflops = peak * (m / 8192) ** 0.5 * (nk / 8192)
                    seconds = metrics["flops"] / (tflops * 1e12)
                    metrics.update(
                        tflops_median=tflops, tflops_best=tflops, timing={"median_ms": seconds * 1e3}
                    )
                records.append(rec("matmul", metrics))
    records.append(
        rec("launch_overhead", {"n_ops": 1000, "eager_us_per_kernel": 6.0, "graph_us_per_kernel": 1.5})
    )
    records.append(
        rec(
            "power",
            {
                "idle": {"mean_w": 16.0, "max_w": 17.0, "n_samples": 10},
                "copy_1gib": None,
                "matmul_bf16_8192": {
                    "mean_w": 70.0,
                    "max_w": 72.0,
                    "mean_sm_clock_mhz": 1020.0,
                    "min_sm_clock_mhz": 990,
                    "n_samples": 10,
                },
            },
        )
    )
    records.append(rec("gpu_info_end", {"telemetry": {"sm_clock_mhz": 990, "temperature_c": 69}}))
    return records
