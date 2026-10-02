"""The cross-milestone summary tables."""

from types import SimpleNamespace

from fastserve.perfmodel.serving import Hardware
from fastserve.report import summary

LARGE, SMALL = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"


def test_versions_puts_the_two_images_side_by_side():
    serving = [
        {"experiment": "serving", "timestamp": "2026-10-01", "env": {"gpu": "NVIDIA L4", "vllm": "0.29.0"}},
        {"experiment": "serving", "timestamp": "2026-10-02", "env": {"gpu": "NVIDIA L4", "vllm": "0.30.0"}},
        {"experiment": "m8_plan_cost", "timestamp": "2026-10-03", "env": {"vllm": "9.9"}},  # not a server
    ]
    research = [
        {"experiment": "probe", "timestamp": "2026-09-29", "env": {"gpu": "NVIDIA L4", "torch": "2.14.0"}}
    ]
    table = summary.versions(serving, research)
    assert "| GPU | NVIDIA L4 | NVIDIA L4 |" in table
    assert "| vLLM | 0.30.0 | — |" in table  # the newest server's, and the research image has no vLLM
    assert "| PyTorch | — | 2.14.0 |" in table
    assert "Triton" not in table  # neither image reported it


def test_key_numbers(m8_records):
    hw = Hardware(bandwidth=250e9, peak={"bf16": 50e12, "fp8": 100e12}, l2_bytes=0)
    cfg = SimpleNamespace(num_params=lambda: 1_000_000_000, kv_bytes_per_token=lambda b: 51_200 * b)
    frozen = {"predictions": [{"model": LARGE, "label": "base", "workload": "m8_latency", "tok_s": 80.0}]}
    # the fixture has only the large model: give the small one the same servers under its name
    both = m8_records + [
        {**r, "metrics": {**r["metrics"], "model": SMALL}} for r in m8_records if r["experiment"] == "serving"
    ]
    table = summary.key_numbers(hw, {LARGE: cfg, SMALL: cfg}, both, frozen)
    assert "| GPU memory bandwidth, read | 250 GB/s | M0 |" in table
    assert "| Peak matmul rate, BF16 · FP8 | 50 · 100 TFLOP/s | M0 |" in table
    assert "| Ridge point, BF16 | 200 FLOPs per byte | M0 |" in table
    assert "| Qwen3-1.7B: weights in BF16 | 2.00 GB (1,000 M parameters) | M1 |" in table
    assert "| Qwen3-1.7B: KV cache per token, BF16 | 100 KiB | M1 |" in table
    assert "| Qwen3-1.7B: one-user ceiling, bandwidth ÷ weight bytes | 125 tokens/s | M1 |" in table
    assert "| Qwen3-1.7B: stock vLLM, one user · 64 users | 100 · 2,000 tokens/s | M8 |" in table
    assert "| Qwen3-1.7B: the full stack, one user | `wkps`, 2.16× stock | M8 |" in table
    assert "| Host time per pass on piecewise CUDA graphs | about 40 ms | M8 |" in table
    assert "| Serving model, frozen: median error · without speculation | 20.0% · 20.0% | M8 |" in table
