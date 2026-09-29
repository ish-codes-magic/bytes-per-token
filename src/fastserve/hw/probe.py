"""M0 hardware probe: measure this GPU's real ceilings and return them as result records.

Everything later is judged against these numbers: the roofline, the performance model, every "% of peak".
Run it in the cloud:  uv run --only-group local modal run infra/modal_app.py::probe
"""

from __future__ import annotations

import time
from typing import Any

from fastserve.hw import bandwidth, matmul, overhead, telemetry
from fastserve.hw.specs import spec_for
from fastserve.results import environment_info, make_record, new_run_id


def run_probe(
    config: dict[str, Any], *, git: dict[str, Any] | None = None, config_path: str | None = None
) -> list[dict[str, Any]]:
    """Run every probe described by `config` (see benchmarks/configs/hw_probe.yaml) on GPU 0."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the hardware probe needs a CUDA GPU")

    run_id = new_run_id()
    env = environment_info()
    spec = spec_for(env.get("gpu", ""))
    records: list[dict[str, Any]] = []

    def add(experiment: str, metrics: dict[str, Any]) -> None:
        records.append(
            make_record(
                experiment, metrics, run_id=run_id, config={"path": config_path, **config}, git=git, env=env
            )
        )

    add("gpu_info", {"spec": spec.to_dict() if spec else None, "telemetry": telemetry.snapshot()})

    bw = config["bandwidth"]
    for result in bandwidth.bandwidth_sweep([2**p for p in bw["sizes_log2"]], iters=bw["iters"]):
        add("bandwidth", result)

    mm = config["matmul"]
    for result in matmul.matmul_sweep(mm["formats"], mm["m"], mm["nk"], iters=mm["iters"]):
        add("matmul", result)

    add("launch_overhead", overhead.launch_overhead(config["launch_overhead"]["n_ops"]))
    add("power", _power_profile(config["power"]["seconds_per_workload"]))

    # Same readings as at the start: lower clocks or higher temperature here would mean throttling.
    add("gpu_info_end", {"telemetry": telemetry.snapshot()})
    return records


def _power_profile(seconds: float) -> dict[str, Any]:
    """Average power while idle, while streaming memory, and while doing dense BF16 math."""
    import torch

    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    workloads = {
        "idle": lambda: time.sleep(0.05),
        "copy_1gib": lambda: dst.copy_(src),
        "matmul_bf16_8192": matmul.make_matmul("bf16", 8192, 8192, 8192),
    }
    return {
        name: telemetry.average_power_during(work, duration_s=seconds) for name, work in workloads.items()
    }
