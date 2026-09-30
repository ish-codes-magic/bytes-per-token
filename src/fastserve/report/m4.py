"""M4 analysis: quantized checkpoints in vLLM → speed, memory, cost and quality tables. Stdlib only.

Records come from results/raw/m4_production.jsonl (servers, task scores, KL, checkpoints). BF16 task scores
come from M2 (results/raw/m2_quality.jsonl): same lm-eval suite, same vLLM, same GPU.
"""

from __future__ import annotations

from typing import Any

from fastserve.report.m2 import plateau
from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]
SMALL, LARGE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"
FORMATS = ("bf16", "fp8", "int8", "gptq", "awq")
FORMAT_LABELS = {
    "bf16": "BF16",
    "fp8": "FP8 W8A8",
    "int8": "INT8 W8A8",
    "gptq": "INT4 W4A16 (GPTQ)",
    "awq": "INT4 W4A16 (AWQ)",
}
DASH = "—"


def _newest(records: Records, experiment: str) -> list[dict[str, Any]]:
    """Metrics of each experiment's records, oldest first (so later ones overwrite in dict building)."""
    return [
        r["metrics"] for r in sorted(records, key=lambda r: r["timestamp"]) if r["experiment"] == experiment
    ]


def serving(records: Records, model: str, fmt: str, workload: str, **load: Any) -> dict[str, Any] | None:
    """The newest serving record for (model, format, workload), optionally matching load keys."""
    found = None
    for m in _newest(records, "serving"):
        same = m["model"] == model and m["server"] == fmt and m["workload"] == workload
        if same and all(m["load"].get(k) == v for k, v in load.items()):
            found = m
    return found


def start(records: Records, model: str, fmt: str) -> dict[str, Any] | None:
    found = None
    for m in _newest(records, "server_start"):
        if m["model"] == model and m["label"] == fmt:
            found = m
    return found


def tpot_b1(records: Records, model: str, fmt: str) -> float | None:
    m = serving(records, model, fmt, "chat")
    return (m["summary"].get("tpot_ms") or {}).get("p50") if m else None


def ttft(records: Records, model: str, fmt: str, workload: str) -> float | None:
    m = serving(records, model, fmt, workload)
    return (m["summary"].get("ttft_ms") or {}).get("p50") if m else None


def decode_tok_s(records: Records, model: str, fmt: str, batch: int) -> float | None:
    m = serving(records, model, fmt, "decode", concurrency=batch)
    return m["summary"].get("output_throughput") if m else None


def saturated(records: Records, model: str, fmt: str) -> dict[str, float] | None:
    """The saturation plateau (M2's definition: requests waiting), from the server's own counters."""
    m = serving(records, model, fmt, "saturation")
    return plateau(m) if m and "server_timeline" in m else None


def tasks(m4: Records, m2_quality: Records, model: str, fmt: str) -> dict[str, float]:
    """Task scores (%) for one model and format; BF16's come from M2."""
    rows = (
        [m for m in _newest(m2_quality, "quality_tasks") if m["model"] == model]
        if fmt == "bf16"
        else [m for m in _newest(m4, "m4_tasks") if m["model"] == model and m["format"] == fmt]
    )
    if not rows:
        return {}
    return {task: 100 * s["score"] for task, s in rows[-1]["scores"].items() if s.get("score") is not None}


def kl(m4: Records, model: str, fmt: str) -> float | None:
    if fmt == "bf16":
        return 0.0
    found = [m for m in _newest(m4, "m3_config") if m["model"] == model and m["config"] == fmt]
    return found[-1]["mean_kl"] if found else None


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def m4_observables(m4: Records, m2_quality: Records, m3: Records) -> dict[str, float | None]:
    """The quantities predicted before M4 (benchmarks/predictions/m4.json)."""
    sat = {(model, fmt): saturated(m4, model, fmt) for model in (SMALL, LARGE) for fmt in FORMATS}

    def sat_ratio(fmt: str) -> float | None:
        a, b = sat[SMALL, fmt], sat[SMALL, "bf16"]
        return _ratio(a and a["output_tok_s"], b and b["output_tok_s"])

    def kv(model: str) -> float | None:
        a, b = start(m4, model, "fp8"), start(m4, model, "bf16")
        return _ratio(a and a.get("kv_cache_tokens"), b and b.get("kv_cache_tokens"))

    def gsm8k_delta(model: str, fmt: str) -> float | None:
        a, b = (
            tasks(m4, m2_quality, model, fmt).get("gsm8k"),
            tasks(m4, m2_quality, model, "bf16").get("gsm8k"),
        )
        return a - b if a is not None and b is not None else None

    ours = [
        m["mean_kl"]
        for m in _newest(m3, "m3_config")
        if m["model"] == SMALL and m["config"] == "w8a8-fp8-token"
    ]
    return {
        "b1_fp8_small": _ratio(tpot_b1(m4, SMALL, "bf16"), tpot_b1(m4, SMALL, "fp8")),
        "b1_int4_small": _ratio(tpot_b1(m4, SMALL, "bf16"), tpot_b1(m4, SMALL, "gptq")),
        "b1_fp8_large": _ratio(tpot_b1(m4, LARGE, "bf16"), tpot_b1(m4, LARGE, "fp8")),
        "b1_int4_large": _ratio(tpot_b1(m4, LARGE, "bf16"), tpot_b1(m4, LARGE, "gptq")),
        "int4_over_fp8_b1": _ratio(decode_tok_s(m4, LARGE, "gptq", 1), decode_tok_s(m4, LARGE, "fp8", 1)),
        "int4_over_fp8_b256": _ratio(
            decode_tok_s(m4, LARGE, "gptq", 256), decode_tok_s(m4, LARGE, "fp8", 256)
        ),
        "awq_over_gptq_speed": _ratio(tpot_b1(m4, LARGE, "gptq"), tpot_b1(m4, LARGE, "awq")),
        "int8_over_fp8_b1": _ratio(tpot_b1(m4, LARGE, "fp8"), tpot_b1(m4, LARGE, "int8")),
        "sat_fp8_small": sat_ratio("fp8"),
        "sat_int4_small": sat_ratio("gptq"),
        "ttft8k_fp8_large": _ratio(ttft(m4, LARGE, "fp8", "long_8k"), ttft(m4, LARGE, "bf16", "long_8k")),
        "ttft8k_int4_large": _ratio(ttft(m4, LARGE, "gptq", "long_8k"), ttft(m4, LARGE, "bf16", "long_8k")),
        "kv_fp8_small": kv(SMALL),
        "kv_fp8_large": kv(LARGE),
        "gsm8k_fp8_small": gsm8k_delta(SMALL, "fp8"),
        "gsm8k_int4_small": gsm8k_delta(SMALL, "gptq"),
        "gsm8k_int4_large": gsm8k_delta(LARGE, "gptq"),
        "kl_fp8_library_over_ours": _ratio(kl(m4, SMALL, "fp8"), ours[-1] if ours else None),
        "kl_int8_library": kl(m4, SMALL, "int8"),
    }


# ---- tables -------------------------------------------------------------------------------------------


def _f(x: float | None, digits: int = 0, suffix: str = "") -> str:
    return DASH if x is None else f"{x:,.{digits}f}{suffix}"


def _kernel(s: dict[str, Any] | None) -> str:
    """The kernel vLLM reported for the quantized linears, shortened (e.g. "Marlin")."""
    names = {
        k.split()[1].removesuffix("LinearKernel").removesuffix("Kernel") for k in (s or {}).get("kernels", [])
    }
    return ", ".join(sorted(names)) or DASH


def speed_table(m4: Records, dollars_per_hour: float = 0.80) -> str:
    """One row per (model, format): memory, batch-1 decode, prefill, saturated capacity and its cost."""
    rows = []
    for model in (SMALL, LARGE):
        base_tpot = tpot_b1(m4, model, "bf16")
        for fmt in FORMATS:
            s, tpot, sat = start(m4, model, fmt), tpot_b1(m4, model, fmt), saturated(m4, model, fmt)
            if s is None and tpot is None:
                continue
            tok_s = sat["output_tok_s"] if sat else None
            rows.append(
                [
                    model.split("/")[-1],
                    FORMAT_LABELS[fmt],
                    _kernel(s) if fmt != "bf16" else DASH,
                    _f(s and s.get("model_memory_gib"), 2),
                    _f(s and s.get("kv_cache_tokens")),
                    _f(tpot, 2),
                    _f(_ratio(base_tpot, tpot), 2, "×"),
                    _f(ttft(m4, model, fmt, "long_8k")),
                    _f(tok_s),
                    _f(dollars_per_hour / (tok_s * 3600) * 1e6 if tok_s else None, 3),
                ]
            )
    headers = [
        "Model",
        "Format",
        "Kernel",
        "Weights (GiB)",
        "KV cache (tokens)",
        "Batch-1 TPOT (ms)",
        "Speedup",
        "TTFT 8k (ms)",
        "Saturated tok/s",
        "$ / 1M tokens",
    ]
    return markdown_table(headers, rows)


def decode_table(m4: Records, model: str, batches: tuple[int, ...] = (1, 4, 16, 64, 256)) -> str:
    """Decode throughput at each batch size, per format, with the speedup over BF16."""
    rows = []
    for batch in batches:
        base = decode_tok_s(m4, model, "bf16", batch)
        row = [str(batch)]
        for fmt in FORMATS:
            tok_s = decode_tok_s(m4, model, fmt, batch)
            ratio = _ratio(tok_s, base)
            row.append(
                DASH if tok_s is None else f"{tok_s:,.0f}" + ("" if fmt == "bf16" else f" ({ratio:.2f}×)")
            )
        rows.append(row)
    return markdown_table(["Batch", *(FORMAT_LABELS[f] for f in FORMATS)], rows)


def quality_table(m4: Records, m2_quality: Records) -> str:
    """KL (nanoserve, the checkpoints' own rounded weights) and task scores (vLLM, real kernels)."""
    rows = []
    for model in (SMALL, LARGE):
        for fmt in FORMATS:
            t, k = tasks(m4, m2_quality, model, fmt), kl(m4, model, fmt)
            if not t and k is None:
                continue
            rows.append(
                [
                    model.split("/")[-1],
                    FORMAT_LABELS[fmt],
                    DASH if k is None else (f"{k:.2e}" if 0 < k < 0.01 else f"{k:.3f}"),
                    *(_f(t.get(task), 1) for task in ("gsm8k", "mmlu", "humaneval")),
                ]
            )
    headers = ["Model", "Format", "KL vs BF16", "GSM8K (%)", "MMLU (%)", "HumanEval pass@1 (%)"]
    return markdown_table(headers, rows)


def checkpoint_table(m4: Records) -> str:
    rows = [
        [
            m["model"].split("/")[-1],
            FORMAT_LABELS[m["format"]],
            f"{m['checkpoint_bytes'] / 1e9:.2f}",
            f"{m['quantize_s']:.0f}",
            str(m["max_levels_per_group"]),
        ]
        for m in _newest(m4, "m4_checkpoint")
    ]
    headers = [
        "Model",
        "Format",
        "Checkpoint (GB)",
        "Quantization time (s)",
        "Max distinct values per 128 weights",
    ]
    return markdown_table(headers, rows)


def fidelity_table(m4: Records) -> str:
    """vLLM's perplexity (real low-bit kernels) against nanoserve's (the same rounded weights, simulated)."""
    served = {(m["model"], m["format"]): m["perplexity"] for m in _newest(m4, "m4_vllm_perplexity")}
    simulated: dict[tuple[str, str], float] = {}
    for m in _newest(m4, "m3_config"):
        simulated[m["model"], m["config"]] = m["perplexity_cand"]
        simulated[m["model"], "bf16"] = m["perplexity_ref"]
    rows = []
    for model in (SMALL, LARGE):
        for fmt in FORMATS:
            a, b = served.get((model, fmt)), simulated.get((model, fmt))
            if a is None and b is None:
                continue
            gap = 100 * (a / b - 1) if a and b else None
            rows.append([model.split("/")[-1], FORMAT_LABELS[fmt], _f(b, 2), _f(a, 2), _f(gap, 1, "%")])
    headers = [
        "Model",
        "Format",
        "Perplexity, nanoserve (simulated)",
        "Perplexity, vLLM (real kernels)",
        "Gap",
    ]
    return markdown_table(headers, rows)
