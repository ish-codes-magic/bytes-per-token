"""Runs only on a GPU machine (L4 via `modal run infra/modal_app.py::test_gpu`)."""

import pytest

from fastserve.hw.analysis import metrics_of
from fastserve.hw.specs import spec_for

pytestmark = pytest.mark.gpu

QUICK_CONFIG = {
    "bandwidth": {"sizes_log2": [20, 28], "iters": 5},
    "matmul": {"formats": ["bf16", "fp8", "int8"], "m": [1, 256], "nk": [1024], "iters": 5},
    "launch_overhead": {"n_ops": 100},
    "power": {"seconds_per_workload": 0.2},
}


def test_read_kernel_really_reads_everything():
    # "If a result looks too good, assume a bug": prove the bandwidth kernel doesn't skip work.
    import torch
    import triton

    from fastserve.hw.bandwidth import _sum_blocks_kernel

    n, block = 10_000, 1024  # deliberately not a multiple of the block size
    x = torch.ones(n, device="cuda")
    grid = (triton.cdiv(n, block),)
    out = torch.empty(grid[0], device="cuda")
    _sum_blocks_kernel()[grid](x, out, n, BLOCK=block)
    assert out.sum().item() == n


def test_cuda_time_returns_one_positive_time_per_iteration():
    import torch

    from fastserve.timing import cuda_time_ms

    x = torch.randn(1024, 1024, device="cuda")
    times = cuda_time_ms(lambda: x @ x, warmup=2, iters=7)
    assert len(times) == 7 and all(t > 0 for t in times)


def test_quick_probe_produces_sane_records():
    from fastserve.hw.probe import run_probe

    records = run_probe(QUICK_CONFIG)
    assert {"gpu_info", "bandwidth", "matmul", "launch_overhead", "power", "gpu_info_end"} <= {
        r["experiment"] for r in records
    }
    assert len({r["run_id"] for r in records}) == 1

    spec = spec_for(records[0]["env"]["gpu"])
    for m in metrics_of(records, "bandwidth"):
        assert m["gbps_median"] > 0
        if spec and m["size_bytes"] >= 4 * spec.l2_bytes:  # a memory-sized transfer can't beat the datasheet
            assert m["gbps_median"] * 1e9 <= 1.05 * spec.bandwidth

    bf16 = [m for m in metrics_of(records, "matmul") if m["format"] == "bf16"]
    assert bf16 and all(m.get("tflops_median", 0) > 0 for m in bf16)

    launch = metrics_of(records, "launch_overhead")[0]
    assert launch["graph_us_per_kernel"] < launch["eager_us_per_kernel"]  # the whole point of CUDA graphs
