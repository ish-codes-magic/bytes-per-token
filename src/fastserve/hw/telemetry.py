"""GPU telemetry via NVML: clocks, temperature and power. Used to spot throttling and to measure energy.

NVML can be missing or blocked inside cloud containers. Every reading then comes back as None, and the
probe records it as missing instead of failing or silently dropping it.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable
from typing import Any


def _nvml():
    """(module, handle for GPU 0), or None if NVML is unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def _read(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # this particular counter isn't exposed here
        return None


def snapshot() -> dict[str, Any] | None:
    """Current clocks, temperature and power. Compare start vs end of a run to detect throttling."""
    nvml = _nvml()
    if nvml is None:
        return None
    m, h = nvml
    power_mw = _read(lambda: m.nvmlDeviceGetPowerUsage(h))
    limit_mw = _read(lambda: m.nvmlDeviceGetEnforcedPowerLimit(h))
    return {
        "sm_clock_mhz": _read(lambda: m.nvmlDeviceGetClockInfo(h, m.NVML_CLOCK_SM)),
        "mem_clock_mhz": _read(lambda: m.nvmlDeviceGetClockInfo(h, m.NVML_CLOCK_MEM)),
        "max_sm_clock_mhz": _read(lambda: m.nvmlDeviceGetMaxClockInfo(h, m.NVML_CLOCK_SM)),
        "max_mem_clock_mhz": _read(lambda: m.nvmlDeviceGetMaxClockInfo(h, m.NVML_CLOCK_MEM)),
        "temperature_c": _read(lambda: m.nvmlDeviceGetTemperature(h, m.NVML_TEMPERATURE_GPU)),
        "power_w": power_mw / 1e3 if power_mw is not None else None,
        "power_limit_w": limit_mw / 1e3 if limit_mw is not None else None,
    }


def average_power_during(
    work: Callable[[], object], *, duration_s: float = 3.0, interval_s: float = 0.02
) -> dict[str, Any] | None:
    """Run `work` repeatedly for `duration_s` seconds while a background thread samples power draw."""
    import torch

    nvml = _nvml()
    if nvml is None or _read(lambda: nvml[0].nvmlDeviceGetPowerUsage(nvml[1])) is None:
        return None
    m, h = nvml

    samples: list[float] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            samples.append(m.nvmlDeviceGetPowerUsage(h) / 1e3)
            time.sleep(interval_s)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        work()
        torch.cuda.synchronize()
    stop.set()
    sampler.join()
    return {"mean_w": statistics.fmean(samples), "max_w": max(samples), "n_samples": len(samples)}
