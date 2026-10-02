"""M7 analysis: the profile, both kernels' microbenchmarks, and nanoserve with the kernels. Stdlib only.

Records come from results/raw/m7_kernels.jsonl. Kernel 1's contenders are timed two ways: `eager` (launched
from Python) and `graph` (replayed from a CUDA graph, as vLLM runs its decode step). `bandwidth` is M0's
measured read bandwidth in bytes/s, as everywhere in the reports.
"""

from __future__ import annotations

from typing import Any

from fastserve.report import m4 as m4_report
from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]
DASH = "—"
NQ_LABELS = {
    "torch": "PyTorch (nanoserve's modules)",
    "vllm-norm": "vLLM `rms_norm`",
    "vllm-quant": "vLLM `scaled_int8_quant`",
    "vllm-separate": "vLLM, both ops",
    "vllm-fused-fp8": "vLLM fused norm + FP8",
    "triton-fused": "Kernel 1",
}
ATT_LABELS = {
    "sdpa-bf16": "PyTorch attention, BF16",
    "flashinfer-fp16": "FlashInfer, FP16",
    "dequant-int8": "Dequantize then attend, INT8",
    "dequant-int4": "Dequantize then attend, INT4",
    "triton-bf16": "Kernel 2, BF16",
    "triton-int8": "Kernel 2, INT8",
    "triton-int4": "Kernel 2, INT4",
}
CACHE_LABELS = {
    "bf16-sdpa": "BF16 + PyTorch attention (before M7)",
    "bf16-kernel": "BF16 + kernel 2",
    "int8-kernel": "INT8 codes + kernel 2",
    "int4-kernel": "INT4 codes + kernel 2",
    "int4-dequant": "INT4 codes, dequantize then attend",
}
LONG = (1, 32768)  # the microbenchmark's long-context shape; nanoserve's is 32,000 (M5's needle length)


def _rows(records: Records, experiment: str, task: str | None = None) -> list[dict[str, Any]]:
    """Metrics of one experiment, oldest first (later ones overwrite), optionally from one task only."""
    found = [r for r in sorted(records, key=lambda r: r["timestamp"]) if r["experiment"] == experiment]
    return [r["metrics"] for r in found if task is None or r["config"].get("task") == task]


def _one(records: Records, experiment: str, task: str | None = None, **match: Any) -> dict[str, Any] | None:
    found = [m for m in _rows(records, experiment, task) if all(m.get(k) == v for k, v in match.items())]
    return found[-1] if found else None


def _f(x: float | None, digits: int = 2, suffix: str = "") -> str:
    return DASH if x is None else f"{x:,.{digits}f}{suffix}"


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def _shape(batch: int, context: int) -> str:
    return f"{batch} × {context:,}"


# ---- the profile -------------------------------------------------------------------------------------------


def profile_table(m7: Records, bandwidth: float) -> str:
    """nanoserve's decode step before M7: attention's share, and how slowly it gets through the cache."""
    rows = []
    for m in sorted(_latest(_rows(m7, "m7_profile"), "batch", "context"), key=lambda m: m["kv_bytes"]):
        rate = m["kv_bytes"] / (m["attention_ms"] / 1e3)  # bytes/s, like `bandwidth`
        rows.append(
            [
                _shape(m["batch"], m["context"]),
                _f(m["step_ms"], 1),
                _f(m["attention_ms"], 1),
                _f(100 * m["attention_share"], 0, "%"),
                f"{m['kernels']:,}",
                _f(m["kv_bytes"] / 1e9, 2),
                _f(rate / 1e9, 0),
                _f(100 * rate / bandwidth, 0, "%"),
            ]
        )
    headers = [
        "Batch × context",
        "Step (ms)",
        "Attention (ms)",
        "Share of step",
        "Kernels launched",
        "KV cache read (GB)",
        "GB/s through attention",
        "Share of M0's bandwidth",
    ]
    return markdown_table(headers, rows)


def _latest(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    """The newest row per key tuple."""
    return list({tuple(m[k] for k in keys): m for m in rows}.values())


def norm_row(m7: Records, rows: int, d: int, task: str | None = None) -> dict[str, Any] | None:
    return _one(m7, "m7_norm_quant", task, rows=rows, d=d)


def norm_ms(m7: Records, rows: int, d: int, name: str, mode: str = "graph", task: str | None = None):
    """A contender's median time in ms (`mode`: eager or graph), or None if absent or failed."""
    row = norm_row(m7, rows, d, task)
    return ((row or {}).get("contenders", {}).get(name, {}).get(mode) or {}).get("ms")


def norm_sizes(m7: Records, task: str | None = None) -> list[tuple[int, int]]:
    return sorted({(m["d"], m["rows"]) for m in _rows(m7, "m7_norm_quant", task)})


def ops_profile_table(m7: Records) -> str:
    """vLLM's norm and INT8-quantization ops as they run in a CUDA graph: what kernel 1 competes with.

    Built from the `ops` task alone, which ran before kernel 1 was timed.
    """
    rows = []
    for d, n in norm_sizes(m7, "ops"):
        row = norm_row(m7, n, d, "ops")
        separate = row["contenders"].get("vllm-separate", {})
        ms = norm_ms(m7, n, d, "vllm-separate", task="ops")
        rows.append(
            [
                f"{n:,} × {d:,}",
                _f(_us(norm_ms(m7, n, d, "vllm-norm", task="ops")), 1),
                _f(_us(norm_ms(m7, n, d, "vllm-quant", task="ops")), 1),
                _f(_us(ms), 1),
                _f(_us(norm_ms(m7, n, d, "vllm-separate", "eager", "ops")), 1),
                _f(_us(norm_ms(m7, n, d, "vllm-fused-fp8", task="ops")), 1),
                _f(separate["bytes_moved"] / (ms / 1e3) / 1e9 if ms else None, 0),
            ]
        )
    headers = [
        "Tokens × width",
        "`rms_norm` (µs)",
        "`scaled_int8_quant` (µs)",
        "Both, in a CUDA graph (µs)",
        "Both, from Python (µs)",
        "vLLM's fused norm + FP8 (µs)",
        "Both: GB/s moved",
    ]
    return markdown_table(headers, rows)


def _us(ms: float | None) -> float | None:
    return None if ms is None else ms * 1e3


# ---- kernel 1 ----------------------------------------------------------------------------------------------


def norm_quant_table(m7: Records, mode: str = "graph") -> str:
    """Kernel 1 against vLLM's two ops, vLLM's fused FP8 op and PyTorch, at every size."""
    rows = []
    for d, n in norm_sizes(m7, "norm_quant"):
        row = norm_row(m7, n, d, "norm_quant")
        ours = norm_ms(m7, n, d, "triton-fused", mode, "norm_quant")
        separate = norm_ms(m7, n, d, "vllm-separate", mode, "norm_quant")
        fused = norm_ms(m7, n, d, "vllm-fused-fp8", mode, "norm_quant")
        moved = row["contenders"].get("triton-fused", {}).get("bytes_moved")
        rows.append(
            [
                f"{n:,} × {d:,}",
                _f(_us(norm_ms(m7, n, d, "torch", mode, "norm_quant")), 1),
                _f(_us(separate), 1),
                _f(_us(fused), 1),
                _f(_us(ours), 1),
                _f(_ratio(separate, ours), 2, "×"),
                _f(_ratio(fused, ours), 2, "×"),
                _f(moved / (ours / 1e3) / 1e9 if ours and moved else None, 0),
            ]
        )
    headers = [
        "Tokens × width",
        "PyTorch (µs)",
        "vLLM, two ops (µs)",
        "vLLM fused FP8 (µs)",
        "**Kernel 1 (µs)**",
        "vs two ops",
        "vs fused FP8",
        "Kernel 1: GB/s moved",
    ]
    return markdown_table(headers, rows)


def norm_exactness_table(m7: Records) -> str:
    rows = []
    for d, n in norm_sizes(m7, "norm_quant"):
        row = norm_row(m7, n, d, "norm_quant")
        ref, vllm = row.get("vs_reference"), row.get("vs_vllm")
        if ref is None:
            continue
        rows.append(
            [
                f"{n:,} × {d:,}",
                _f(100 * (1 - ref["codes_differing"]), 3, "%"),
                str(ref["max_code_diff"]),
                _f(100 * (1 - vllm["codes_differing"]), 2, "%") if vllm else DASH,
                str(vllm["max_code_diff"]) if vllm else DASH,
            ]
        )
    headers = [
        "Tokens × width",
        "Codes equal to the reference",
        "Largest difference (codes)",
        "Codes equal to vLLM's two ops",
        "Largest difference (codes)",
    ]
    return markdown_table(headers, rows)


def norm_warps_table(m7: Records) -> str:
    cells = _latest(_rows(m7, "m7_norm_quant_warps"), "rows", "d", "num_warps")
    warps = sorted({m["num_warps"] for m in cells})
    rows = []
    for n, d in sorted({(m["rows"], m["d"]) for m in cells}):
        by = {m["num_warps"]: m.get("ms") for m in cells if (m["rows"], m["d"]) == (n, d)}
        best = min((w for w in warps if by.get(w)), key=lambda w: by[w])
        rows.append([f"{n:,} × {d:,}"] + [_f(_us(by.get(w)), 1) + (" ←" if w == best else "") for w in warps])
    return markdown_table(["Tokens × width"] + [f"{w} warps (µs)" for w in warps], rows)


def m4_gap(m4: Records, cfg: Any, bandwidth: float, model: str) -> dict[str, float] | None:
    """M4's bytes-only model for W8A8-INT8 at batch 1: predicted and measured TPOT, and the unexplained ms."""
    measured = {}
    for fmt in ("bf16", "int8"):
        m = m4_report.serving(m4, model, fmt, "decode", concurrency=1)
        measured[fmt] = m["summary"]["tpot_ms"]["p50"] if m else None
    if None in measured.values():
        return None
    overhead = measured["bf16"] - 1e3 * m4_report.step_bytes(cfg, "bf16") / bandwidth
    predicted = 1e3 * m4_report.step_bytes(cfg, "int8") / bandwidth + overhead
    return {
        "predicted_ms": predicted,
        "measured_ms": measured["int8"],
        "gap_ms": measured["int8"] - predicted,
    }


def quant_launch_cost(m7: Records, cfg: Any) -> dict[str, float] | None:
    """What a W8A8-INT8 decode step spends on separate quantization ops: 4 per layer (the inputs of qkv, o,
    gate/up and down), each one launch of `scaled_int8_quant` on a single token."""
    us = _us(norm_ms(m7, 1, cfg.hidden_size, "vllm-quant", "graph"))
    if us is None:
        return None
    return {"launches": 4 * cfg.num_layers, "us_each": us, "ms": 4 * cfg.num_layers * us / 1e3}


def m4_gap_table(m7: Records, m4: Records, configs: dict[str, Any], bandwidth: float) -> str:
    """Does the cost of the separate quantization launches account for M4's W8A8-INT8 gap?"""
    rows = []
    for model, cfg in configs.items():
        gap, cost = m4_gap(m4, cfg, bandwidth, model), quant_launch_cost(m7, cfg)
        if gap is None or cost is None:
            continue
        fusable = cost["ms"] / 2  # 2 of the 4 follow an RMSNorm
        rows.append(
            [
                model.split("/")[-1],
                _f(gap["predicted_ms"], 2),
                _f(gap["measured_ms"], 2),
                _f(gap["gap_ms"], 2),
                f"{cost['launches']} × {cost['us_each']:.2f} µs",
                _f(cost["ms"], 2),
                _f(100 * cost["ms"] / gap["gap_ms"], 0, "%"),
                _f(100 * fusable / gap["measured_ms"], 1, "%"),
            ]
        )
    headers = [
        "Model",
        "TPOT from bytes (ms)",
        "Measured TPOT (ms)",
        "Unexplained (ms)",
        "Quantization launches per step",
        "Their cost (ms)",
        "Share of the gap",
        "Fusable with a norm: share of TPOT",
    ]
    return markdown_table(headers, rows)


# ---- kernel 2 ----------------------------------------------------------------------------------------------


def attention_cases(m7: Records) -> list[dict[str, Any]]:
    rows = [m for m in _rows(m7, "m7_attention") if "contenders" in m]
    return sorted(_latest(rows, "batch", "context"), key=lambda m: (m["batch"], m["context"]))


def att(m7: Records, batch: int, context: int, name: str, key: str = "ms") -> float | None:
    row = _one(m7, "m7_attention", batch=batch, context=context)
    return ((row or {}).get("contenders", {}).get(name) or {}).get(key)


def attention_table(m7: Records) -> str:
    """One layer's decode attention: every contender's time, and kernel 2's INT4 speedups."""
    names = ["sdpa-bf16", "flashinfer-fp16", "dequant-int4", "triton-bf16", "triton-int8", "triton-int4"]
    rows = []
    for m in attention_cases(m7):
        ms = {name: (m["contenders"].get(name) or {}).get("ms") for name in names}
        rows.append(
            [_shape(m["batch"], m["context"])]
            + [_f(_us(ms[name]), 0) for name in names]
            + [
                _f(_ratio(ms["sdpa-bf16"], ms["triton-int4"]), 2, "×"),
                _f(_ratio(ms["flashinfer-fp16"], ms["triton-int4"]), 2, "×"),
            ]
        )
    headers = (
        ["Batch × context"]
        + [
            ("**" + ATT_LABELS[n] + "**" if n.startswith("triton") else ATT_LABELS[n]) + " (µs)"
            for n in names
        ]
        + ["INT4 kernel vs PyTorch", "INT4 kernel vs FlashInfer"]
    )
    return markdown_table(headers, rows)


def roofline_table(m7: Records, bandwidth: float) -> str:
    """Each contender's cache bytes ÷ time, as a share of the measured memory bandwidth."""
    names = ["sdpa-bf16", "flashinfer-fp16", "triton-bf16", "triton-int8", "triton-int4"]
    rows = []
    for m in attention_cases(m7):
        cells = [_shape(m["batch"], m["context"])]
        cells.append(_f((m["contenders"].get("triton-int4") or {}).get("cache_bytes", 0) / 1e6, 1))
        for name in names:
            gbps = (m["contenders"].get(name) or {}).get("gbps")
            cells.append(DASH if gbps is None else f"{gbps:,.0f} ({100 * gbps * 1e9 / bandwidth:.0f}%)")
        rows.append(cells)
    headers = ["Batch × context", "INT4 cache (MB)"] + [ATT_LABELS[n] + ": GB/s" for n in names]
    return markdown_table(headers, rows)


def attention_error_table(m7: Records) -> str:
    """Kernel vs reference on the same codes, and each format's distance from unquantized attention."""
    rows = []
    for m in attention_cases(m7):
        c = m["contenders"]
        if "error_vs_reference" not in (c.get("triton-int4") or {}):
            continue
        rows.append(
            [_shape(m["batch"], m["context"])]
            + [
                _e((c.get(f"triton-{fmt}") or {}).get("error_vs_reference"))
                for fmt in ("bf16", "int8", "int4")
            ]
            + [
                _e((c.get(name) or {}).get("error_vs_bf16"))
                for name in ("flashinfer-fp16", "triton-int8", "triton-int4")
            ]
        )
    headers = [
        "Batch × context",
        "Kernel vs reference, BF16",
        "INT8",
        "INT4",
        "vs unquantized attention: FlashInfer",
        "INT8 codes",
        "INT4 codes",
    ]
    return markdown_table(headers, rows)


def _e(x: float | None) -> str:
    return DASH if x is None else f"{x:.1e}"


def worst_kernel_error(m7: Records, name: str = "triton-int4") -> float | None:
    errors = [(m["contenders"].get(name) or {}).get("error_vs_reference") for m in attention_cases(m7)]
    errors = [e for e in errors if e is not None]
    return max(errors) if errors else None


def tune_cells(m7: Records, batch: int, context: int, bits: int) -> list[dict[str, Any]]:
    rows = [
        m
        for m in _rows(m7, "m7_tune")
        if (m["batch"], m["context"], m["bits"]) == (batch, context, bits) and "ms" in m
    ]
    return _latest(rows, "split", "num_warps")


def tune_best(m7: Records, batch: int, context: int, bits: int) -> dict[str, Any] | None:
    cells = tune_cells(m7, batch, context, bits)
    return min(cells, key=lambda m: m["ms"]) if cells else None


def tune_table(m7: Records) -> str:
    """Per shape and format: the best (split, warps), and what the two obvious wrong choices cost."""
    shapes = sorted({(m["batch"], m["context"], m["bits"]) for m in _rows(m7, "m7_tune")})
    rows = []
    for batch, context, bits in shapes:
        cells, best = tune_cells(m7, batch, context, bits), tune_best(m7, batch, context, bits)
        if best is None:
            continue
        whole = min((m["ms"] for m in cells if m["split"] == context), default=None)  # no splitting
        finest = min((m["ms"] for m in cells if m["split"] == min(c["split"] for c in cells)), default=None)
        rows.append(
            [
                _shape(batch, context),
                "BF16" if bits == 16 else f"INT{bits}",
                f"{best['split']:,}",
                str(best["num_warps"]),
                f"{best['programs']:,}",
                _f(_us(best["ms"]), 0),
                _f(_ratio(whole, best["ms"]), 1, "×"),
                _f(_ratio(finest, best["ms"]), 1, "×"),
                _f(_ratio(max(m["ms"] for m in cells), best["ms"]), 1, "×"),
            ]
        )
    headers = [
        "Batch × context",
        "Format",
        "Best tokens per program",
        "Warps",
        "Programs",
        "Time (µs)",
        "No splitting ÷ best",
        "One key group per program ÷ best",
        "Worst ÷ best",
    ]
    return markdown_table(headers, rows)


# ---- nanoserve ---------------------------------------------------------------------------------------------


def nanoserve_cases(m7: Records) -> list[tuple[int, int]]:
    return sorted({(m["batch"], m["context"]) for m in _rows(m7, "m7_nanoserve")}, key=lambda s: s[0] * s[1])


def step(m7: Records, cache: str, batch: int, context: int) -> dict[str, Any] | None:
    return _one(m7, "m7_nanoserve", cache=cache, batch=batch, context=context)


def nanoserve_table(m7: Records, dollars_per_hour: float) -> str:
    """The real model's decode step through each cache: time, speedup, cache size, cost, and logit drift."""
    rows = []
    for batch, context in nanoserve_cases(m7):
        base = step(m7, "bf16-sdpa", batch, context)
        for cache, label in CACHE_LABELS.items():
            m = step(m7, cache, batch, context)
            if m is None:
                continue
            rows.append(
                [
                    _shape(batch, context),
                    label,
                    _f(m["step_ms"], 1),
                    _f(_ratio(base["step_ms"], m["step_ms"]) if base else None, 2, "×"),
                    _f(m["kv_bytes"] / 2**20, 0),
                    _f(dollars_per_hour / (m["tokens_per_s"] * 3600) * 1e6, 2),
                    _f(m.get("kl_first_step"), 4),
                    _f(100 * m["top1_agreement"], 0, "%") if "top1_agreement" in m else DASH,
                ]
            )
    headers = [
        "Batch × context",
        "KV cache and reader",
        "Step (ms)",
        "vs before M7",
        "KV cache (MiB)",
        "$ per 1M tokens",
        "KL of the next token vs BF16",
        "Same top token",
    ]
    return markdown_table(headers, rows)


def quality_table(m7: Records, m5: Records | None = None) -> str:
    """KL and needle recall through the real codes and kernel, next to M5's simulation where it has one."""
    simulated = {"int8-kernel": "int8-kivi", "int4-kernel": "int4-kivi"}
    rows = []
    for m in _latest(_rows(m7, "m7_kl"), "cache"):
        needle = _one(m7, "m7_needle", cache=m["cache"])
        sim_kl = sim_needle = None
        if m5 is not None and m["cache"] in simulated:
            sim = _one(m5, "m5_kv_kl", model=m["model"], policy=simulated[m["cache"]])
            sim_kl = sim["mean_kl"] if sim else None
            found = _one(m5, "m5_kv_needle", model=m["model"], policy=simulated[m["cache"]])
            sim_needle = found["pass_rate"] if found else None
        rows.append(
            [
                CACHE_LABELS[m["cache"]],
                f"{m['mean_kl']:.2g}",
                _f(100 * m["top1_agreement"], 1, "%"),
                _f(m["perplexity_cand"], 2),
                _f(m["perplexity_ref"], 2),
                DASH if sim_kl is None else f"{sim_kl:.2g}",
                _f(100 * needle["pass_rate"], 0, "%") if needle else DASH,
                _f(100 * sim_needle, 0, "%") if sim_needle is not None else DASH,
            ]
        )
    headers = [
        "KV cache and reader",
        "KL vs BF16 cache (nats)",
        "Same top token",
        "Perplexity",
        "BF16 perplexity",
        "M5's simulated KL",
        "Needle recall",
        "M5's simulated recall",
    ]
    return markdown_table(headers, rows)


def norm_in_model_table(m7: Records) -> str:
    m = _one(m7, "m7_norm_in_model")
    if m is None:
        return DASH
    rows = [
        ["Reference (norm, then quantize, in PyTorch)", _f(m["reference_step_ms"], 1), DASH, DASH],
        [
            "Kernel 1",
            _f(m["fused_step_ms"], 1),
            _f(m["max_logit_diff"], 3),
            _f(100 * m["top1_agreement"], 0, "%"),
        ],
    ]
    headers = [
        "Pre-norms emit INT8 through",
        "Decode step (ms)",
        "Largest logit difference",
        "Same top token",
    ]
    return markdown_table(headers, rows)


def production_context_table(m7: Records, m5: Records, cfg: Any, vllm_context: int = 32768) -> str:
    """nanoserve's 32k-token decode step next to vLLM's on the same model and GPU.

    vLLM's rows are M5's `long_32k` workload (one user, `vllm_context`-token prompts).
    """
    rows = []
    for cache, label in CACHE_LABELS.items():
        m = step(m7, cache, 1, 32000)
        if m is not None:
            rows.append(["nanoserve", label, _f(m["step_ms"], 1), _f(m["kv_bytes"] / 2**30, 2)])
    vllm = {
        "bf16kv": "BF16 KV, FlashAttention",
        "bf16kv-flashinfer": "BF16 KV, FlashInfer",
        "fp8kv": "FP8 KV",
    }
    for server, label in vllm.items():
        found = None
        for m in _rows(m5, "serving"):
            if (m["model"], m["server"], m["workload"]) == ("Qwen/Qwen3-0.6B", server, "long_32k"):
                found = m
        if found:
            tpot = (found["summary"].get("tpot_ms") or {}).get("p50")
            kv_bytes = cfg.kv_bytes_per_token(1 if server == "fp8kv" else 2) * vllm_context
            rows.append(["vLLM 0.30", label, _f(tpot, 1), _f(kv_bytes / 2**30, 2)])
    return markdown_table(
        ["Engine", "KV cache and attention", "Decode step (ms)", "KV cache read (GiB)"], rows
    )


# ---- predictions -------------------------------------------------------------------------------------------


def m7_observables(m7: Records, bandwidth: float) -> dict[str, float | None]:
    """The measured value of every quantity in benchmarks/predictions/m7.json."""
    d = 2048

    def nq(rows: int, name: str, mode: str = "graph") -> float | None:
        return norm_ms(m7, rows, d, name, mode, "norm_quant")

    big = norm_row(m7, 32768, d, "norm_quant") or {}
    moved = (big.get("contenders", {}).get("triton-fused") or {}).get("bytes_moved")
    exact = [m for m in _rows(m7, "m7_norm_quant", "norm_quant") if "vs_reference" in m]

    def a(name: str, shape: tuple[int, int] = LONG, key: str = "ms") -> float | None:
        return att(m7, *shape, name, key)

    best = tune_best(m7, *LONG, 4)
    whole = min((m["ms"] for m in tune_cells(m7, *LONG, 4) if m["split"] == LONG[1]), default=None)

    def ns(cache: str, batch: int, context: int) -> float | None:
        base, m = step(m7, "bf16-sdpa", batch, context), step(m7, cache, batch, context)
        return _ratio(base["step_ms"], m["step_ms"]) if base and m else None

    int4, bf16 = step(m7, "int4-kernel", 1, 32000), step(m7, "bf16-sdpa", 1, 32000)

    def kl(cache: str) -> float | None:
        m = _one(m7, "m7_kl", cache=cache)
        return m["mean_kl"] if m else None

    needle = _one(m7, "m7_needle", cache="int4-kernel")
    return {
        "nq_graph_1": _ratio(nq(1, "vllm-separate"), nq(1, "triton-fused")),
        "nq_graph_4096": _ratio(nq(4096, "vllm-separate"), nq(4096, "triton-fused")),
        "nq_graph_32768": _ratio(nq(32768, "vllm-separate"), nq(32768, "triton-fused")),
        "nq_bandwidth": _ratio(moved, (nq(32768, "triton-fused") or 0) / 1e3 * bandwidth) if moved else None,
        "nq_vs_fused_fp8": _ratio(nq(32768, "vllm-fused-fp8"), nq(32768, "triton-fused")),
        "nq_eager_1": _ratio(nq(1, "vllm-separate", "eager"), nq(1, "triton-fused", "eager")),
        "nq_codes_reference": min((1 - m["vs_reference"]["codes_differing"] for m in exact), default=None),
        "nq_codes_vllm": min(
            (1 - m["vs_vllm"]["codes_differing"] for m in exact if "vs_vllm" in m), default=None
        ),
        "att_bf16_kernel_vs_sdpa_32k": _ratio(a("sdpa-bf16"), a("triton-bf16")),
        "att_int4_vs_bf16_kernel_32k": _ratio(a("triton-bf16"), a("triton-int4")),
        "att_int8_vs_bf16_kernel_32k": _ratio(a("triton-bf16"), a("triton-int8")),
        "att_bf16_kernel_vs_flashinfer_32k": _ratio(a("flashinfer-fp16"), a("triton-bf16")),
        "att_int4_vs_flashinfer_32k": _ratio(a("flashinfer-fp16"), a("triton-int4")),
        "att_int4_vs_dequant_32k": _ratio(a("dequant-int4"), a("triton-int4")),
        "att_bf16_bandwidth_32k": _ratio(a("triton-bf16", key="gbps"), bandwidth / 1e9),
        "att_int4_bandwidth_32k": _ratio(a("triton-int4", key="gbps"), bandwidth / 1e9),
        "att_int4_vs_sdpa_512": _ratio(a("sdpa-bf16", (1, 512)), a("triton-int4", (1, 512))),
        "att_int4_error": worst_kernel_error(m7),
        "tune_best_split_32k": best["split"] if best else None,
        "tune_nosplit_penalty": _ratio(whole, best["ms"]) if best else None,
        "ns_int4_512": ns("int4-kernel", 1, 512),
        "ns_int4_32k": ns("int4-kernel", 1, 32000),
        "ns_int4_b8": ns("int4-kernel", 8, 4096),
        "ns_dequant_32k": ns("int4-dequant", 1, 32000),
        "ns_memory": _ratio(int4["kv_bytes_per_token"], bf16["kv_bytes_per_token"])
        if int4 and bf16
        else None,
        "kl_bf16_kernel": kl("bf16-kernel"),
        "kl_int8": kl("int8-kernel"),
        "kl_int4": kl("int4-kernel"),
        "needle_int4": needle["pass_rate"] if needle else None,
    }
