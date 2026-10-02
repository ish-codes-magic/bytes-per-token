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

    saturation: 200 running, KV cache half full, 4,000 tokens/s in 50 ms steps, a queue from t = 1 s to 9 s;
    then the queue is gone: 100 running, KV a quarter full, 25 ms steps, no new prompts, until t = 14 s.
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
        saturated = [[t, 200, 10 if t >= 1 else 0, 0.5, 1000 * t, 4000 * t, 20 * t, 0] for t in range(10)]
        saturated += [[t, 100, 0, 0.25, 10_000, 4000 * t, 200 + 40 * (t - 10), 0] for t in range(10, 15)]
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


@pytest.fixture
def m3_records() -> list[dict[str, Any]]:
    """A hand-made M3 campaign: each configuration's KL is a round number, so every ratio is easy to check."""
    env, git = {"gpu": "NVIDIA L4"}, {"commit": "0123456789abcdef", "dirty": False}
    kls = {
        "bf16": 0.0,
        "rtn-int8-channel": 0.001,
        "rtn-int4-channel": 0.5,
        "rtn-int4-g128": 0.1,
        "rtn-int4-g64": 0.08,
        "rtn-int4-g128-full": 0.09,
        "rtn-int3-g128": 0.6,
        "rtn-int2-g64-asym": 5.0,
        "nf4-b64": 0.06,
        "fp8-weight-channel": 0.004,
        "gptq-int4-g128": 0.045,
        "awq-int4-g128": 0.054,
        "library-gptq-int4-g128": 0.05,
        "library-awq-int4-g128": 0.06,
        "rot-rtn-int4-channel": 0.2,
        "w8a8-fp8-token": 0.01,
        "w8a8-int8-tensor-static": 2.0,
        "sq-w8a8-int8-tensor-static": 0.2,
        "calib-c4-8": 0.054,
        "calib-code": 0.0495,
        "calib-wikitext": 0.0405,
    }
    entries = {
        "bf16": {"method": "bf16"},
        "rtn-int4-g128": {"method": "rtn", "bits": 4},
        "gptq-int4-g128": {"method": "gptq", "bits": 4, "full_range": True},
        "w8a8-fp8-token": {"method": "w8a8", "format": "fp8", "act": "token"},
    }
    records = []

    def add(experiment, metrics, task="grids", stamp="2026-09-30T00:00:00+00:00"):
        record = make_record(experiment, metrics, run_id="m3test", env=env, git=git, config={"task": task})
        record["timestamp"] = stamp
        records.append(record)

    for model, scale in (("Qwen/Qwen3-0.6B", 1.0), ("Qwen/Qwen3-1.7B", 0.5)):
        for name, kl in kls.items():
            entry = {"name": name, **entries.get(name, {"method": "rtn", "bits": 4})}
            metrics = {
                "model": model,
                "config": name,
                "entry": entry,
                "bits_per_weight": 4.125,
                "model_gb": 0.5,
                "mean_kl": kl * scale,
                "top1_agreement": 1 - kl / 10,
                "perplexity_ref": 20.0,
                "perplexity_cand": 20.0 * (1 + kl),
            }
            add("m3_config", metrics)
    # a stale, older result for one configuration: the newest must win
    stale = dict(records[3]["metrics"], mean_kl=9.9)
    add("m3_config", stale, stamp="2026-09-29T00:00:00+00:00")
    add("m3_outliers", {"model": "Qwen/Qwen3-0.6B", "residual_ratio": 1500.0})
    cells = [
        {"layer": layer, "module": module, "kl": 0.004 if module == "down_proj" else 0.001}
        for layer in range(28)
        for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    ]
    add("m3_sensitivity", {"model": "Qwen/Qwen3-0.6B", "spec": "INT4 g128 sym", "windows": 4, "cells": cells})
    return records


@pytest.fixture
def m4_records() -> list[dict[str, Any]]:
    """A hand-made M4 campaign: every format of both models, with round numbers."""
    env = {"gpu": "NVIDIA L4", "gpu_memory_bytes": 24 * 2**30}
    git = {"commit": "0123456789abcdef", "dirty": False}
    speed = {"bf16": 1.0, "fp8": 1.5, "int8": 1.4, "gptq": 2.0, "awq": 2.0}  # batch-1 speedups
    at_256 = {"bf16": 1.0, "fp8": 1.2, "int8": 1.1, "gptq": 0.9, "awq": 0.9}
    weights = {"bf16": 1.2, "fp8": 0.8, "int8": 0.8, "gptq": 0.6, "awq": 0.6}
    records = []

    def add(experiment, metrics, stamp="2026-10-01T00:00:00+00:00"):
        record = make_record(experiment, metrics, run_id="m4test", env=env, git=git, config={})
        record["timestamp"] = stamp
        records.append(record)

    def summary(tok_s, tpot, ttft=20.0):
        return {"output_throughput": tok_s, "tpot_ms": {"p50": tpot}, "ttft_ms": {"p50": ttft}}

    for model, scale in (("Qwen/Qwen3-0.6B", 1.0), ("Qwen/Qwen3-1.7B", 2.5)):
        for fmt in speed:
            kernels = [] if fmt == "bf16" else [f"Using {fmt.title()}LinearKernel for CompressedTensorsW"]
            add(
                "server_start",
                {
                    "label": fmt,
                    "model": model,
                    "kv_cache_tokens": int(100_000 * (1 + 0.1 * (2 - weights[fmt]))),
                    "model_memory_gib": weights[fmt] * scale,
                    "kv_cache_memory_gib": 19.0,
                    "kernels": kernels,
                },
            )
            base = {"server": fmt, "model": model, "requests": {"columns": [], "rows": []}}
            tpot = 6.0 * scale / speed[fmt]
            add(
                "serving",
                {
                    **base,
                    "workload": "chat",
                    "load": {"mode": "closed", "concurrency": 1},
                    "summary": summary(1000 / tpot, tpot),
                },
            )
            add(
                "serving",
                {
                    **base,
                    "workload": "long_8k",
                    "load": {"mode": "closed", "concurrency": 1},
                    "summary": summary(50, 10, ttft=400 * scale / (1.5 if fmt in ("fp8", "int8") else 1)),
                },
            )
            for batch in (1, 4, 16, 64, 256):
                ratio = at_256[fmt] if batch == 256 else speed[fmt]  # INT4 leads until batch 256
                add(
                    "serving",
                    {
                        **base,
                        "workload": "decode",
                        "load": {"mode": "closed", "concurrency": batch},
                        "summary": summary(100 * batch * ratio / scale, 10),
                    },
                )
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
            tok_s = 2000 * at_256[fmt] / scale
            rows = [[t, 200, 10, 0.9, 500 * t, tok_s * t, 10 * t, 0] for t in range(5)]
            add(
                "serving",
                {
                    **base,
                    "workload": "saturation",
                    "load": {"mode": "closed", "concurrency": 512},
                    "summary": summary(tok_s, 100),
                    "server_timeline": {"columns": columns, "rows": rows},
                },
            )
            int4 = fmt in ("gptq", "awq")  # 200 answers: INT4 loses 10 correct ones to loops
            add(
                "m4_gsm8k",
                {
                    "model": model,
                    "format": fmt,
                    "n": 200,
                    "strict_accuracy": 0.4,
                    "buckets": {
                        "correct": 80 - 10 * int4,
                        "right, then kept talking": 0,
                        "wrong answer": 100,
                        "looping": 10 * int4,
                        "no final answer": 20,
                    },
                    "talks_past_answer": 0.1 + 0.2 * int4,
                    "mean_words": 120.0 + 30 * int4,
                    "examples": {
                        "looping": ["and again and again"] if int4 else [],
                        "wrong answer": ["#### 5"],
                    },
                },
            )
            if fmt != "bf16":
                add(
                    "m4_tasks",
                    {
                        "model": model,
                        "format": fmt,
                        "scores": {"gsm8k": {"score": 0.40 - 0.05 * (fmt in ("gptq", "awq"))}},
                    },
                )
                kl = 0.02 if fmt in ("fp8", "int8") else 0.3
                add(
                    "m3_config",
                    {
                        "model": model,
                        "config": fmt,
                        "mean_kl": kl,
                        "perplexity_ref": 20.0,
                        "perplexity_cand": 20.0 * (1 + kl),
                    },
                )
                add(
                    "m4_vllm_perplexity",
                    {"model": model, "format": fmt, "perplexity": 20.0 * (1 + kl) * 1.01},
                )
                add(
                    "m4_checkpoint",
                    {
                        "model": model,
                        "format": fmt,
                        "checkpoint": f"/cache/m4/x-{fmt}",
                        "quantize_s": 60.0,
                        "checkpoint_bytes": int(weights[fmt] * 1e9),
                        "max_levels_per_group": 16,
                    },
                )
    return records


M5_POLICIES = [
    {"name": "bf16"},
    {"name": "fp8", "keys": {"kind": "fp8"}, "values": {"kind": "fp8"}},
    {"name": "int4-token", "keys": {"bits": 4}, "values": {"bits": 4}},
    {"name": "int4-kivi", "keys": {"bits": 4, "axis": "channel", "group": 32}, "values": {"bits": 4}},
    {"name": "streaming", "sinks": 4, "window": 1020},
]


@pytest.fixture
def m5_policies() -> list[dict[str, Any]]:
    return M5_POLICIES


def _needle_cells(rule) -> list[dict[str, Any]]:
    return [
        {"length": n, "depth": d, "secret": "1", "passed": rule(n, d), "answer": "1"}
        for n in (1024, 32000)
        for d in (0.0, 1.0)
    ]


@pytest.fixture
def m5_records() -> list[dict[str, Any]]:
    """A hand-made M5 campaign with round numbers: FP8 KV doubles capacity, caching cuts TTFT 4×."""
    env = {"gpu": "NVIDIA L4", "gpu_memory_bytes": 24 * 2**30}
    records = []

    def add(experiment, metrics):
        record = make_record(
            experiment, metrics, run_id="m5test", env=env, git={"commit": "0" * 16}, config={}
        )
        record["timestamp"] = "2026-10-01T00:00:00+00:00"
        records.append(record)

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
    kv = {"bf16kv": 1.0, "bf16kv-flashinfer": 1.0, "fp8kv": 2.0, "fp8w-fp8kv": 2.0}  # capacity multiplier
    speed = {"bf16kv": 1.0, "bf16kv-flashinfer": 1.1, "fp8kv": 1.5, "fp8w-fp8kv": 1.6}
    for model, scale in (("Qwen/Qwen3-0.6B", 1.0), ("Qwen/Qwen3-1.7B", 2.0)):
        for label in kv:
            add(
                "server_start",
                {
                    "label": label,
                    "model": model,
                    "kv_cache_tokens": int(170_000 * kv[label]),
                    "kv_cache_memory_gib": 18.0,
                    "model_memory_gib": 1.0,
                    "max_concurrency": 4.0 * kv[label],
                    "attention_backend": "FLASH_ATTN" if label == "bf16kv" else "FLASHINFER",
                },
            )
            for workload, running, tok_s in (("saturation", 250, 2000), ("capacity", 40 * kv[label], 400)):
                rate = tok_s * speed[label] / scale
                rows = [[t, running, 10, 0.9, 1000 * t, rate * t, 20 * t, 0] for t in range(5)]
                add(
                    "serving",
                    {
                        "server": label,
                        "model": model,
                        "workload": workload,
                        "load": {},
                        "summary": {"output_throughput": rate},
                        "server_timeline": {"columns": columns, "rows": rows},
                    },
                )
            add(
                "serving",
                {
                    "server": label,
                    "model": model,
                    "workload": "long_32k",
                    "load": {},
                    "summary": {"tpot_ms": {"p50": 20.0 / speed[label]}, "ttft_ms": {"p50": 3000.0}},
                },
            )
        request_cols = ["id", "prompt_len", "output_tokens", "scheduled", "sent", "first_token", "finished"]
        request_cols.append("cached_tokens")
        for label, cached, ttft in (("prefix-off", 0, 0.2), ("prefix-on", 1600, 0.05)):
            rows = [
                [i, 2000, 64, 0.0, i * 1.0, i * 1.0 + ttft, i * 1.0 + 1.0, cached if i else 0]
                for i in range(4)
            ]
            for workload in ("multi_turn", "shared_prefix"):
                summary = {
                    "output_throughput": 500.0 * (1.5 if cached else 1.0) / scale,
                    "ttft_ms": {"p50": 1e3 * ttft},
                    "tpot_ms": {"p50": 10.0},
                }
                add(
                    "serving",
                    {
                        "server": label,
                        "model": model,
                        "workload": workload,
                        "load": {},
                        "summary": summary,
                        "requests": {"columns": request_cols, "rows": rows},
                    },
                )
        for policy, value in (("bf16", 0.0), ("fp8", 0.01), ("int4-token", 2.0), ("int4-kivi", 0.04)):
            add(
                "m5_kv_kl",
                {"model": model, "policy": policy, "mean_kl": value * scale, "top1_agreement": 0.9},
            )
        add("m5_kv_kl", {"model": model, "policy": "streaming", "mean_kl": 0.05, "top1_agreement": 0.9})
        add("m5_vllm_perplexity", {"model": model, "kv_cache_dtype": "fp8", "perplexity": 20.2})
    for policy, rule in (("bf16", lambda n, d: True), ("streaming", lambda n, d: d == 1.0)):
        cells = _needle_cells(rule)
        rate = sum(c["passed"] for c in cells) / len(cells)
        add("m5_kv_needle", {"model": "Qwen/Qwen3-0.6B", "policy": policy, "cells": cells, "pass_rate": rate})
    cells = _needle_cells(lambda n, d: True)
    add(
        "m5_vllm_needle",
        {"model": "Qwen/Qwen3-0.6B", "kv_cache_dtype": "fp8", "cells": cells, "pass_rate": 1.0},
    )
    keys = [[1.0] * 7 + [40.0] for _ in range(2)]  # 2 heads × 8 channels, one outlier key channel
    add(
        "m5_kv_stats",
        {
            "model": "Qwen/Qwen3-0.6B",
            "key_ratio": [40.0, 8.0, 8.0],
            "value_ratio": [2.0, 2.0, 2.0],
            "profiles": {"14": {"keys": keys, "values": [[1.0] * 8 for _ in range(2)]}},
        },
    )
    return records


@pytest.fixture
def m6_records() -> list[dict[str, Any]]:
    """A hand-made M6 campaign: a drafter that is right 60% of the time, costly at k = 5 and at 64 users."""
    from fastserve.spec.simulate import expected_tokens

    records = []

    def add(experiment, metrics):
        record = make_record(
            experiment, metrics, run_id="m6test", env={}, git={"commit": "0" * 16}, config={}
        )
        record["timestamp"] = "2026-10-01T00:00:00+00:00"
        records.append(record)

    tasks = ("chat", "code", "math", "summarize")
    agree = {"chat": 0.5, "code": 0.7, "math": 0.6, "summarize": 0.6, "all": 0.6}
    for target, factor in (("bf16", 1.0), ("fp8", 1.0), ("awq", 0.9)):
        for drafter in ("small", "small-awq", "ngram") if target == "bf16" else ("small",):
            for task, a in agree.items():
                a = a * factor * (0.9 if drafter == "small-awq" else 1.0)
                by_k = {
                    str(k): {
                        "tokens_per_round": 1.2 if drafter == "ngram" else expected_tokens(a, k),
                        "acceptance_rate": a,
                        "accepted_histogram": [40, 30, 20, 10][: k + 1],
                    }
                    for k in range(1, 9)
                }
                row = {
                    "target": target,
                    "drafter": drafter,
                    "task": task,
                    "prompts": 16,
                    "tokens": 3000,
                    "by_k": by_k,
                }
                if drafter != "ngram":
                    row["rates"] = {"agree": a, "after_agree": a + 0.1, "after_miss": a - 0.1}
                add("m6_agreement", row)
    for drafter in ("small", "ngram"):
        for task in tasks:
            pieces = ["def", " f", "(", "x", "):", "\n", "    return", " x"]
            flags = [True, True, False, True, True, False, drafter == "small", False]
            shown = {"target": "bf16", "drafter": drafter, "task": task, "k": 4}
            add("m6_highlight", {**shown, "pieces": pieces, "from_draft": flags})
    for dtype, same in (("bfloat16", 6), ("float32", 8)):
        add(
            "m6_loop_check",
            {
                "target": "bf16",
                "drafter": "small",
                "dtype": dtype,
                "prompts": 8,
                "k": 4,
                "identical_outputs": same,
                "rounds_match_replay": same,
                "first_divergence": [],
                "tokens_per_round_actual": 2.5,
                "tokens_per_round_replay": 2.5,
            },
        )
    for task in tasks:
        prefixes = [{"tokens": t, "tv_spec_vs_plain": 0.1 * t, "tv_plain_vs_plain": 0.1 * t} for t in (1, 2)]
        stats = {
            "samples": 1000,
            "k": 3,
            "chi_square": 50.0,
            "chi_square_plain": 45.0,
            "dof": 40,
            "limit": 84.7,
        }
        add(
            "m6_lossless", {"target": "bf16", "drafter": "small", "task": task, **stats, "prefixes": prefixes}
        )
    add(
        "m6_step_times",
        {"role": "target", "model": "t", "context": 600, "ms": {"1": 20.0, "4": 22.0, "9": 24.0}},
    )
    add("m6_step_times", {"role": "draft", "model": "d", "context": 600, "ms": {"1": 10.0}})

    # vLLM: one user per task, then mixed tasks at 4, 16, 64 users. Speedups by (method, k) and users.
    one_user = {
        "none": 1.0,
        "draft-k1": 1.1,
        "draft-k3": 1.2,
        "draft-k5": 0.9,
        "ngram-k3": 1.05,
        "eagle3-k3": 1.8,
    }
    busy = {4: 0.9, 16: 0.7, 64: 0.5}  # multiplies the one-user speedup of every speculative method
    columns = ["t", "spec_drafts", "spec_draft_tokens", "spec_accepted"]
    cols = ["id", "prompt_len", "output_tokens", "scheduled", "sent", "first_token", "finished", "output_crc"]
    for target in ("bf16", "fp8", "awq"):
        for method, gain in one_user.items():
            if target != "bf16" and method not in ("none", "draft-k3"):
                continue
            label = f"{target}-{method}"
            k = int(method[-1]) if method != "none" else 0
            loads = [(f"spec_{t}", 1, 100.0 * gain) for t in tasks]
            loads += [("spec_mixed", u, 100.0 * u * (gain * busy[u] if k else 1.0)) for u in busy]
            for workload, users, rate in loads:
                rows = [[i, 50, 100, 0.0, 0.0, 0.1, 1.0, 7 if (k and i == 0) else 1] for i in range(4)]
                metrics = {
                    "server": label,
                    "model": "Qwen/Qwen3-1.7B",
                    "workload": workload,
                    "load": {"mode": "closed", "concurrency": users},
                    "summary": {"output_throughput": rate},
                    "requests": {"columns": cols, "rows": rows},
                }
                if k:  # 100 rounds of k draft tokens, 60% accepted
                    timeline = {"columns": columns, "rows": [[0, 0, 0, 0], [1, 100, 100 * k, 60 * k]]}
                    timeline["spec_accepted_per_position"] = [60.0] * k
                    metrics["server_timeline"] = timeline
                add("serving", metrics)
    return records


@pytest.fixture
def m7_records() -> list[dict[str, Any]]:
    """A hand-made M7 campaign with round ratios: kernel 1 twice as fast as two ops, INT4 10× PyTorch."""
    records = []

    def add(experiment, metrics, task=""):
        record = make_record(
            experiment, metrics, run_id="m7test", env={}, git={"commit": "0" * 16}, config={"task": task}
        )
        record["timestamp"] = "2026-10-02T00:00:00+00:00"
        records.append(record)

    for context, step, attention in ((512, 40.0, 4.0), (32000, 140.0, 100.0)):
        profile = {"batch": 1, "context": context, "step_ms": step, "attention_ms": attention}
        profile.update(attention_share=attention / step, kernels=2000, kv_bytes=1e5 * context)
        add("m7_profile", profile)

    def timed(ms):
        return {"ms": ms, "timing": {"median_ms": ms}}

    # rows → (two ops, fused FP8, kernel 1) in ms, replayed from a CUDA graph; eager adds a launch cost
    sizes = {1: (0.004, 0.003, 0.002), 4096: (0.04, 0.16, 0.02), 32768: (2.0, 1.6, 0.8)}
    for task in ("ops", "norm_quant"):
        for rows, (separate, fused_fp8, ours) in sizes.items():
            moved = rows * (2048 * 2 + 2048 + 4)
            contenders = {
                "torch": {"bytes_moved": None, "eager": timed(0.3), "graph": timed(0.03)},
                "vllm-norm": {"bytes_moved": 1, "eager": timed(0.02), "graph": timed(separate / 2)},
                "vllm-quant": {"bytes_moved": 1, "eager": timed(0.04), "graph": timed(separate / 2)},
                "vllm-separate": {
                    "bytes_moved": rows * 2048 * 7,
                    "eager": timed(separate + 0.056),
                    "graph": timed(separate),
                },
                "vllm-fused-fp8": {"bytes_moved": moved, "eager": timed(0.05), "graph": timed(fused_fp8)},
            }
            row = {"rows": rows, "d": 2048, "dtype": "bfloat16", "contenders": contenders}
            if task == "norm_quant":
                contenders["triton-fused"] = {
                    "bytes_moved": moved,
                    "eager": timed(ours + 0.028),
                    "graph": timed(ours),
                }
                row["vs_reference"] = {"max_code_diff": 1, "codes_differing": 0.0002 if rows > 1 else 0.0}
                row["vs_vllm"] = {"max_code_diff": 1, "codes_differing": 0.04}
            add("m7_norm_quant", row, task)
    for rows in (256, 32768):
        for warps, ms in ((1, 0.9), (4, 0.3), (8, 0.6)):
            warp_row = {"rows": rows, "d": 2048, "num_warps": warps, **timed(ms)}
            add("m7_norm_quant_warps", warp_row, "norm_quant")

    # One layer's attention. Bytes per token and KV head: BF16 512, INT8 276, INT4 148 (8 KV heads).
    shapes = {(1, 512): 0.1, (1, 32768): 1.0, (16, 2048): 1.0}  # a time scale per shape
    for (batch, context), unit in shapes.items():
        tokens = batch * context * 8
        times = {
            "sdpa-bf16": (3.5, 512),
            "flashinfer-fp16": (0.7, 512),
            "dequant-int4": (35.0, 148),
            "triton-bf16": (1.0, 512),
            "triton-int8": (0.7, 276),
            "triton-int4": (0.35, 148),
        }
        contenders = {}
        for name, (ms, per_token) in times.items():
            if name.startswith("flashinfer") and batch != 1:
                continue
            ms, nbytes = ms * unit, tokens * per_token
            contenders[name] = {**timed(ms), "cache_bytes": nbytes, "gbps": nbytes / (ms / 1e3) / 1e9}
            contenders[name].update(eager_ms=ms + 0.2, graph_ms=ms / 2)  # Python adds 200 µs; L2 halves it
            contenders[name]["write_flush_ms"] = ms + 0.13  # evicting modified lines: a constant
            if name.startswith("triton"):
                contenders[name].update(error_vs_reference=2e-5, error_vs_bf16=0.01, split=512)
        if context == 512:  # launch-bound: the kernel loses
            contenders["triton-int4"]["ms"] = 0.5
        add("m7_attention", {"batch": batch, "context": context, "contenders": contenders}, "attention")
    add("m7_attention", {"batch": 64, "context": 32768, "skipped": "too many tokens"}, "attention")

    grid = ((32, 1, 1.4), (32, 4, 0.7), (512, 1, 0.5), (512, 4, 0.35), (32768, 1, 4.0), (32768, 4, 3.5))
    for split, warps, ms in grid:
        tune = {"batch": 1, "context": 32768, "bits": 4, "split": split, "num_warps": warps}
        add("m7_tune", {**tune, "programs": 8 * (32768 // split), "cache_bytes": 1, **timed(ms)}, "tune")

    steps = {
        (1, 512): {"bf16-sdpa": 45.0, "int4-kernel": 50.0},
        (1, 32000): {
            "bf16-sdpa": 140.0,
            "bf16-kernel": 70.0,
            "int8-kernel": 65.0,
            "int4-kernel": 56.0,
            "int4-dequant": 700.0,
        },
        (8, 4096): {"bf16-sdpa": 128.0, "int4-kernel": 64.0},
    }
    for (batch, context), caches in steps.items():
        for cache, ms in caches.items():
            per_token = 114688 if cache.startswith("bf16") else (65408 if "int8" in cache else 33152)
            row = {"model": "Qwen/Qwen3-0.6B", "cache": cache, "batch": batch, "context": context}
            row.update(step_ms=ms, tokens_per_s=batch / (ms / 1e3), kv_bytes_per_token=per_token)
            row["kv_bytes"] = per_token * batch * context
            if cache != "bf16-sdpa":
                row.update(top1_agreement=1.0, max_logit_diff=0.5, kl_first_step=0.01)
            add("m7_nanoserve", row, "nanoserve")
    for cache, busy in (("bf16-sdpa", 40), ("int4-kernel", 5)):
        events = [[0, 0, 10_000], [1, 20_000, busy * 1000], [0, (30 + busy) * 1000, 10_000]]  # µs
        names = ["gemv", "decode_attention_kernel" if cache == "int4-kernel" else "fmha_cutlassF_bf16"]
        timeline = {"cache": cache, "batch": 1, "context": 16384, "names": names, "events": events}
        add("m7_timeline", timeline, "nanoserve")
    norm = {"batch": 1, "context": 128, "reference_step_ms": 60.0, "fused_step_ms": 50.0}
    add("m7_norm_in_model", {**norm, "max_logit_diff": 0.01, "top1_agreement": 1.0}, "nanoserve")

    for cache, kl in (("bf16-kernel", 1e-5), ("int8-kernel", 0.001), ("int4-kernel", 0.02)):
        row = {"model": "Qwen/Qwen3-0.6B", "cache": cache, "window": 512, "positions": 2044, "mean_kl": kl}
        row.update(top1_agreement=1 - kl, perplexity_ref=20.0, perplexity_cand=20.0 + kl)
        add("m7_kl", row, "quality")
    cells = [{"length": 1024, "depth": 0.5, "secret": "1", "passed": True, "answer": "1"}] * 4
    needle = {"model": "Qwen/Qwen3-0.6B", "cache": "int4-kernel", "cells": cells, "pass_rate": 1.0}
    add("m7_needle", needle, "quality")
    return records
