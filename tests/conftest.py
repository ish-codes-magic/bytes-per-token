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


# -- nanoserve: tiny random-weight Qwen3 models (no downloads) -----------------------------------------------

TINY_QWEN3 = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,  # GQA: 2 query heads per KV head
    "head_dim": 16,
    "max_position_embeddings": 256,
    "initializer_range": 0.2,  # larger than default so logits are decisive (no near-ties in greedy tests)
}


def tiny_qwen3_pair(tie_word_embeddings: bool = False, seed: int = 0):
    """(Hugging Face model, nanoserve model) with identical random weights, float32 on CPU."""
    import torch
    import transformers

    from fastserve.engine.config import ModelConfig
    from fastserve.engine.loader import from_state_dict

    torch.manual_seed(seed)
    hf_config = transformers.Qwen3Config(**TINY_QWEN3, tie_word_embeddings=tie_word_embeddings)
    hf = transformers.Qwen3ForCausalLM._from_config(hf_config, attn_implementation="eager").eval()
    ours = from_state_dict(
        ModelConfig.from_hf(hf_config.to_dict()), hf.state_dict(), device="cpu", dtype=torch.float32
    )
    return hf, ours


@pytest.fixture
def tiny_model():
    pytest.importorskip("transformers")
    return tiny_qwen3_pair()[1]


@pytest.fixture
def make_tiny_qwen3():
    """Factory fixture: make_tiny_qwen3(tie_word_embeddings=...) -> (hf_model, nanoserve_model)."""
    pytest.importorskip("transformers")
    return tiny_qwen3_pair


@pytest.fixture
def m1_records() -> list[dict[str, Any]]:
    """A small hand-made M1 run, shaped like the real one."""
    env = {"gpu": "NVIDIA L4", "cpu": "x"}
    git = {"commit": "0123456789abcdef", "dirty": False}

    def rec(experiment: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return make_record(
            experiment, metrics, run_id="m1test", env=env, git=git, config={"model": "Qwen/Qwen3-0.6B"}
        )

    records = []
    for impl, diff, top1 in (("eager", 0.0, 1.0), ("sdpa", 0.05, 0.9)):
        for prompt, tokens in (("The capital of France is", 5), ("def f(x):\n    return", 7)):
            metrics = {
                "hf_attention": impl,
                "prompt": prompt,
                "tokens": tokens,
                "max_abs_diff": diff,
                "mean_abs_diff": diff / 10,
                "max_abs_logit": 20.0,
                "top1_agreement": top1,
                "mean_kl_hf_to_ours": diff / 100,
            }
            records.append(rec("hf_parity", metrics))
    for batch, ms in ((1, 50.0), (16, 52.0), (64, 54.0)):
        speed = {"batch": batch, "prompt_len": 128, "steps": 64, "step_ms_median": ms}
        records.append(rec("decode_speed", {**speed, "tokens_per_s": batch / (ms / 1e3)}))
    for length, ms in ((512, 52.0), (2048, 360.0)):
        records.append(rec("prefill_speed", {"length": length, "ms_median": ms}))
    matmul = {"matmul": {"count": 253, "ms": 5.4}, "elementwise": {"count": 1000, "ms": 1.6}}
    records.append(
        rec(
            "kernel_profile",
            {"label": "decode, batch 1", "kernels": 2000, "kernel_ms": 8.5, "by_category": matmul},
        )
    )
    records.append(
        rec(
            "kernel_profile",
            {"label": "prefill, 512 tokens", "kernels": 2200, "kernel_ms": 26.0, "by_category": matmul},
        )
    )
    parts = {
        "embedding": 0.2,
        "RMSNorm": 14.0,
        "attention block": 46.0,
        "MLP": 7.0,
        "LM head": 1.3,
        "other (RoPE tables, sampling, Python)": 7.5,
    }
    for label in ("decode, batch 1", "prefill, 512 tokens"):
        records.append(
            rec("component_times", {"label": label, "total_ms": sum(parts.values()), "components_ms": parts})
        )
    maps = {str(layer): [[1.0, 0.0], [0.7, 0.3]] for layer in (0, 1)}
    records.append(
        rec("attention_maps", {"tokens": 2, "head": 0, "maps": maps, "sink_by_layer": [0.1, 0.8, 0.6]})
    )
    log = [
        {
            "step": 0,
            "running": [0, 1],
            "waiting": [2],
            "block_tables": {"0": [0, 1], "1": [2]},
            "free_blocks": 5,
        },
        {
            "step": 1,
            "running": [1, 2],
            "waiting": [],
            "block_tables": {"1": [2], "2": [0, 3]},
            "free_blocks": 4,
        },
    ]
    batching = {
        "num_blocks": 8,
        "block_size": 16,
        "max_batch": 2,
        "steps": 2,
        "tokens_generated": 10,
        "log": log,
    }
    records.append(rec("continuous_batching", batching))
    return records


@pytest.fixture
def m2_records() -> list[dict[str, Any]]:
    """A small hand-made M2 run: two models, every workload, a few load points."""
    env, git = {"gpu": "NVIDIA L4"}, {"commit": "0123456789abcdef", "dirty": False}

    def rec(experiment: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return make_record(experiment, metrics, run_id="m2test", env=env, git=git, config={})

    def summary(tok_s, ttft, tpot, good=1.0, req_s=1.0):
        dist = lambda x: {"mean": x, "p50": x, "p90": 1.5 * x, "p99": 2 * x, "max": 3 * x}  # noqa: E731
        return {
            "requests": 10,
            "completed": 10,
            "errors": 0,
            "duration_s": 10.0,
            "request_throughput": req_s,
            "output_throughput": tok_s,
            "goodput_requests": good * req_s,
            "goodput_fraction": good,
            "ttft_ms": dist(ttft),
            "tpot_ms": dist(tpot),
            "itl_ms": dist(tpot),
            "e2e_ms": dist(10 * tpot),
            "dollars_per_million_output_tokens": 0.8 / (tok_s * 3600) * 1e6,
        }

    def requests(n=10, gap=0.1, ttft=0.02, dur=1.0):
        rows = [
            [i, 100, 50, i * gap, i * gap, i * gap + ttft * (1 + i), i * gap + ttft * (1 + i) + dur]
            for i in range(n)
        ]
        return {
            "columns": ["id", "prompt_len", "output_tokens", "scheduled", "sent", "first_token", "finished"],
            "rows": rows,
        }

    records = []
    for model, scale in (("Qwen/Qwen3-0.6B", 1.0), ("Qwen/Qwen3-1.7B", 2.5)):
        records.append(
            rec(
                "server_start",
                {"model": model, "label": "vllm-bf16", "startup_s": 60, "kv_cache_tokens": 170000},
            )
        )
        records.append(
            rec(
                "serving",
                {
                    "model": model,
                    "workload": "chat",
                    "load": {"mode": "closed", "concurrency": 1},
                    "summary": summary(150 / scale, 20, 6.5 * scale),
                    "requests": requests(),
                },
            )
        )
        for rate, tok_s, good in ((2, 500, 1.0), (8, 2000, 0.95), (16, 3500, 0.5)):
            records.append(
                rec(
                    "serving",
                    {
                        "model": model,
                        "workload": "throughput",
                        "load": {"mode": "open", "rate": rate},
                        "summary": summary(tok_s / scale, 30 * rate, 7 + rate, good, rate * 0.9),
                        "requests": requests(ttft=0.01 * rate),
                    },
                )
            )
        records.append(
            rec(
                "serving",
                {
                    "model": model,
                    "workload": "throughput",
                    "load": {"mode": "closed", "concurrency": 64},
                    "summary": summary(4000 / scale, 200, 20),
                    "requests": requests(),
                },
            )
        )
        for workload, ttft, tpot in (("long_8k", 150, 9), ("long_16k", 320, 13), ("long_32k", 700, 20)):
            records.append(
                rec(
                    "serving",
                    {
                        "model": model,
                        "workload": workload,
                        "load": {"mode": "closed", "concurrency": 1},
                        "summary": summary(50, ttft * scale, tpot * scale),
                        "requests": requests(n=3),
                    },
                )
            )
        records.append(
            rec(
                "serving",
                {
                    "model": model,
                    "workload": "shared_prefix",
                    "load": {"mode": "open", "rate": 16},
                    "summary": summary(900, 80, 9, 1.0, 14.0),
                    "requests": requests(),
                },
            )
        )
    return records


@pytest.fixture
def m2_saturation_records() -> list[dict[str, Any]]:
    """A hand-made saturation run: server timelines with round numbers, so every rate is easy to check.

    saturation: 200 running, KV cache half full, 4,000 tokens/s in 50 ms steps, a queue from t = 1 s to 9 s.
    shared_prefix: 60 running, 100 ms steps each carrying 1,988 prompt tokens, a queue throughout.
    """
    env, git = {"gpu": "NVIDIA L4"}, {"commit": "0123456789abcdef", "dirty": False}
    columns = [
        "t",
        "running",
        "waiting",
        "kv_usage",
        "prompt_tokens",
        "generation_tokens",
        "steps",
        "preemptions",
    ]

    def rec(experiment: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return make_record(experiment, metrics, run_id="m2sat", env=env, git=git, config={})

    def requests(n: int, prompt: int, output: int) -> dict[str, Any]:
        rows = [[i, prompt, output, 0.1 * i, 0.1 * i, 0.1 * i + 0.05, 0.1 * i + 1.05] for i in range(n)]
        return {
            "columns": ["id", "prompt_len", "output_tokens", "scheduled", "sent", "first_token", "finished"],
            "rows": rows,
        }

    records = []
    for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
        start = {
            "model": model,
            "label": "vllm-bf16",
            "kv_cache_tokens": 100_000,
        }
        records.append(rec("server_start", start))
        saturated = [
            [t, 200, 10 if 1 <= t <= 9 else 0, 0.5, 1000 * t, 4000 * t, 20 * t, 0] for t in range(11)
        ]
        records.append(
            rec(
                "serving",
                {
                    "model": model,
                    "workload": "saturation",
                    "load": {"mode": "closed", "concurrency": 512},
                    "summary": {"output_throughput": 3000.0},
                    "requests": requests(20, 256, 192),
                    "server_timeline": {"columns": columns, "rows": saturated},
                },
            )
        )
        shared = [[t, 60, 5, 0.9, 19_880 * t, 600 * t, 10 * t, 0] for t in range(5)]
        records.append(
            rec(
                "serving",
                {
                    "model": model,
                    "workload": "shared_prefix",
                    "load": {"mode": "open", "rate": 16},
                    "summary": {"output_throughput": 500.0},
                    "requests": requests(20, 2112, 64),
                    "server_timeline": {"columns": columns, "rows": shared},
                },
            )
        )
    return records
