"""M8: self times from a profiler trace."""

import pytest

pytest.importorskip("torch")

from fastserve.experiments.m8 import self_times, summarize_trace  # noqa: E402


def event(name, cat, ts, dur, tid=1):
    return {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur, "pid": 1, "tid": tid}


EVENTS = [
    event("step", "user_annotation", 0, 1000),
    event("build", "python_function", 100, 600),
    event("aten::copy_", "cpu_op", 200, 100),
    event("cudaMemcpyAsync", "cuda_runtime", 220, 50),
    event("aten::copy_", "cpu_op", 400, 100),
    event("attention", "kernel", 300, 2000, tid=99),  # the GPU's own timeline
    {"ph": "i", "name": "marker", "ts": 5},  # not a complete event: ignored
]


def test_self_time_is_duration_minus_children():
    timed = {(e["name"], e["ts"]): e["self_us"] for e in self_times(EVENTS)}
    assert timed[("step", 0)] == 400  # 1000 − build's 600
    assert timed[("build", 100)] == 400  # 600 − two copies of 100
    assert timed[("aten::copy_", 200)] == 50 and timed[("aten::copy_", 400)] == 100
    assert timed[("attention", 300)] == 2000  # another thread: nobody's child


def test_summary_separates_host_from_gpu():
    summary = summarize_trace(EVENTS)
    assert summary["host_self_ms"] == pytest.approx(1.0)  # the step's 1,000 µs, split among its parts
    assert summary["gpu_ms"] == pytest.approx(2.0)
    assert summary["span_ms"] == pytest.approx(2.3)
    top = summary["host"][0]
    assert {top["name"], summary["host"][1]["name"]} == {"step", "build"} and top["self_ms"] == 0.4
    copies = next(r for r in summary["host"] if r["name"] == "aten::copy_")
    assert copies["calls"] == 2 and copies["self_ms"] == pytest.approx(0.15)
    assert copies["total_ms"] == pytest.approx(0.2)
    assert [r["name"] for r in summary["gpu"]] == ["attention"]
    assert [r["name"] for r in summary["annotations"]] == ["step"]
