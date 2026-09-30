"""M4 figures: quantized checkpoints in vLLM. Each returns (figure, caption computed from the data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fastserve.report.m4 import (
    FORMAT_LABELS,
    FORMATS,
    LARGE,
    SMALL,
    decode_tok_s,
    saturated,
    serving,
)
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, apply_style, save_figure

Records = list[dict[str, Any]]
BATCHES = (1, 4, 16, 64, 256)
# BF16 is always the gray baseline; each format keeps its color everywhere (style.SERIES uses the same idea)
STYLE = {
    "bf16": (BASELINE_GRAY, "o"),
    "fp8": (OKABE_ITO["blue"], "s"),
    "int8": (OKABE_ITO["sky_blue"], "D"),
    "gptq": (OKABE_ITO["orange"], "^"),
    "awq": (OKABE_ITO["vermillion"], "v"),
}
BYTES_PER_WEIGHT = {"bf16": 2.0, "fp8": 1.0, "int8": 1.0, "gptq": 4.125 / 8, "awq": 4.125 / 8}


def speedup_vs_batch(m4: Records) -> tuple[plt.Figure, str]:
    """Decode throughput relative to BF16 at each batch size, per format; the INT4/FP8 crossover marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    crossovers = {}
    for ax, model in zip(axes, (SMALL, LARGE), strict=True):
        for fmt in FORMATS:
            color, marker = STYLE[fmt]
            points = [
                (b, decode_tok_s(m4, model, fmt, b) / decode_tok_s(m4, model, "bf16", b))
                for b in BATCHES
                if decode_tok_s(m4, model, fmt, b) and decode_tok_s(m4, model, "bf16", b)
            ]
            if points:
                ax.plot(*zip(*points, strict=True), color=color, marker=marker, label=FORMAT_LABELS[fmt])
        fp8 = [decode_tok_s(m4, model, "fp8", b) for b in BATCHES]
        int4 = [decode_tok_s(m4, model, "gptq", b) for b in BATCHES]
        crossing = next((b for b, f, i in zip(BATCHES, fp8, int4, strict=True) if f and i and f >= i), None)
        crossovers[model] = crossing
        if crossing:
            ax.axvline(crossing, color=BASELINE_GRAY, ls=":", lw=1)
            ax.annotate("FP8 overtakes INT4", (crossing, ax.get_ylim()[1]), fontsize=8, ha="right", va="top")
        ax.set(xscale="log", xlabel="decode batch (concurrent sequences)", title=model.split("/")[-1])
        ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
    axes[0].set_ylabel("decode throughput ÷ BF16")
    axes[0].legend(fontsize=8)
    b1 = decode_tok_s(m4, LARGE, "gptq", 1) / decode_tok_s(m4, LARGE, "bf16", 1)
    b256 = decode_tok_s(m4, LARGE, "gptq", 256) / decode_tok_s(m4, LARGE, "bf16", 256)
    where = (
        f"FP8 overtakes it from batch {crossovers[LARGE]}"
        if crossovers.get(LARGE)
        else "it stays ahead of FP8 at every batch measured"
    )
    caption = (
        f"On Qwen3-1.7B, INT4 decodes {b1:.1f}× faster than BF16 at batch 1 but {b256:.1f}× at batch 256; "
        f"{where}."
    )
    return fig, caption


def roofline(m4: Records, configs: dict[str, Any], hw: dict[str, float]) -> tuple[plt.Figure, str]:
    """Qwen3-1.7B's decode steps on the L4's measured roofline: fewer bytes move each format right."""
    cfg = configs[LARGE]
    head = cfg.vocab_size * cfg.hidden_size  # the tied LM head, read in BF16 every step
    linear = cfg.num_params() - head
    kv_per_token, context = cfg.kv_bytes_per_token(), 128 + 128  # decode workload: 128 prompt + ~half of 256
    fig, ax = plt.subplots(figsize=(8, 5))
    intensity = np.logspace(0, 3.5, 200)
    ax.plot(
        intensity, hw["bandwidth"] * intensity / 1e12, color="black", lw=1, label="memory bound (measured)"
    )
    for name, peak in (("BF16", hw["bf16"]), ("FP8", hw["fp8"])):
        ax.axhline(peak / 1e12, color=BASELINE_GRAY, ls="--", lw=1)
        ax.annotate(f"{name} peak", (intensity[0], peak / 1e12), fontsize=8, va="bottom")
    best = {}
    for fmt in FORMATS:
        color, marker = STYLE[fmt]
        points = []
        for batch in BATCHES:
            m = serving(m4, LARGE, fmt, "decode", concurrency=batch)
            tpot = (m or {}).get("summary", {}).get("tpot_ms", {}).get("p50") if m else None
            if not tpot:
                continue
            flops = 2 * (linear + head) * batch
            moved = linear * BYTES_PER_WEIGHT[fmt] + 2 * head + batch * context * kv_per_token
            points.append((flops / moved, flops / (tpot / 1e3) / 1e12))
        if points:
            ax.plot(*zip(*points, strict=True), color=color, marker=marker, label=FORMAT_LABELS[fmt])
            best[fmt] = max(p[1] for p in points)
    ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOPs per byte moved)", ylabel="TFLOP/s")
    ax.legend(fontsize=8)
    caption = (
        f"Smaller weights move every format's decode points right; at batch 256 FP8 reaches "
        f"{best.get('fp8', 0):.1f} TFLOP/s against BF16's {best.get('bf16', 0):.1f}, far below the peaks: "
        "the KV cache and fixed costs keep decode memory-bound."
    )
    return fig, caption


def memory_budget(m4_records: Records) -> tuple[plt.Figure, str]:
    """Where each server's GPU memory goes: weights, KV cache, and everything else vLLM reserves."""
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for model in (SMALL, LARGE):
        for fmt in FORMATS:
            found = [
                r
                for r in sorted(m4_records, key=lambda r: r["timestamp"])
                if r["experiment"] == "server_start"
                and r["metrics"]["model"] == model
                and r["metrics"]["label"] == fmt
            ]
            if not found or found[-1]["metrics"].get("model_memory_gib") is None:
                continue
            m, env = found[-1]["metrics"], found[-1]["env"]
            total = 0.9 * env.get("gpu_memory_bytes", 0) / 2**30  # vLLM's default gpu_memory_utilization
            kv = m.get("kv_cache_memory_gib") or 0.0
            rows.append((f"{model.split('/')[-1]} {FORMAT_LABELS[fmt]}", m["model_memory_gib"], kv, total))
    labels = [r[0] for r in rows]
    weights = np.array([r[1] for r in rows])
    kv = np.array([r[2] for r in rows])
    other = np.maximum(np.array([r[3] for r in rows]) - weights - kv, 0)
    ax.barh(labels, weights, color=OKABE_ITO["orange"], label="weights")
    ax.barh(labels, kv, left=weights, color=OKABE_ITO["blue"], label="KV cache")
    ax.barh(
        labels,
        other,
        left=weights + kv,
        color=BASELINE_GRAY,
        alpha=0.5,
        label="activations, graphs, workspace",
    )
    ax.invert_yaxis()
    ax.set(xlabel="GPU memory (GiB)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3, frameon=False)
    freed = {r[0]: r[2] for r in rows}
    small = [r for r in rows if r[0].startswith("Qwen3-1.7B")]
    caption = "Smaller weights hand their memory to the KV cache"
    if small:
        base = next((r for r in small if r[0].endswith("BF16")), None)
        int4 = next((r for r in small if "GPTQ" in r[0]), None)
        if base and int4:
            caption += (
                f": on Qwen3-1.7B, INT4 weights free {base[1] - int4[1]:.1f} GiB and the KV cache grows from "
                f"{freed[base[0]]:.1f} to {freed[int4[0]]:.1f} GiB"
            )
    return fig, caption + "."


def _costs(m4: Records, model: str, regime: str, dollars_per_hour: float) -> dict[str, float]:
    """$ per 1M output tokens per format: one user (the chat workload) or the saturated server."""
    costs = {}
    for fmt in FORMATS:
        if regime == "one user":
            m = serving(m4, model, fmt, "chat")
            tok_s = m["summary"].get("output_throughput") if m else None
        else:
            sat = saturated(m4, model, fmt)
            tok_s = sat["output_tok_s"] if sat else None
        if tok_s:
            costs[fmt] = dollars_per_hour / (tok_s * 3600) * 1e6
    return costs


def waterfall(m4: Records, dollars_per_hour: float = 0.80) -> tuple[plt.Figure, str]:
    """Waterfall v1: $ per 1M output tokens, BF16 then each format as a change from it, for one user and for
    a saturated server."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    best: dict[tuple[str, str], tuple[str, float]] = {}
    for row, regime in enumerate(("one user", "saturated")):
        for col, model in enumerate((SMALL, LARGE)):
            ax, costs = axes[row, col], _costs(m4, model, regime, dollars_per_hour)
            if "bf16" not in costs:
                continue
            base, names = costs["bf16"], list(costs)
            for i, fmt in enumerate(names):
                if fmt == "bf16":
                    ax.bar(i, base, color=BASELINE_GRAY)
                    continue
                delta = costs[fmt] - base
                ax.bar(
                    i, delta, bottom=base, color=OKABE_ITO["green"] if delta < 0 else OKABE_ITO["vermillion"]
                )
                ax.annotate(
                    f"{delta / base:+.0%}", (i, max(base, costs[fmt])), ha="center", va="bottom", fontsize=8
                )
            ax.set_xticks(
                range(len(names)), [FORMAT_LABELS[f].replace(" (", "\n(") for f in names], fontsize=8
            )
            ax.set(title=f"{model.split('/')[-1]}, {regime}", ylabel="$ per 1M output tokens")
            ax.set_ylim(0, max(costs.values()) * 1.15)
            cheapest = min(costs, key=costs.get)
            best[model, regime] = (FORMAT_LABELS[cheapest], costs[cheapest] / base - 1)
    fig.tight_layout()
    one, sat = best.get((LARGE, "one user")), best.get((LARGE, "saturated"))
    if not one or not sat:
        return fig, "Not every regime was measured."
    caption = (
        f"On Qwen3-1.7B the cheapest format cuts $ per 1M tokens by {-one[1]:.0%} for one user ({one[0]}) "
        f"but by {-sat[1]:.0%} on a saturated server ({sat[0]}), where the KV cache dominates every step."
    )
    return fig, caption


def make_all(m4: Records, configs: dict[str, Any], hw: dict[str, float], out_dir: str | Path) -> list[Path]:
    apply_style()
    figures = {
        "m4_speedup_vs_batch": lambda: speedup_vs_batch(m4),
        "m4_roofline": lambda: roofline(m4, configs, hw),
        "m4_memory_budget": lambda: memory_budget(m4),
        "m4_waterfall": lambda: waterfall(m4),
    }
    written: list[Path] = []
    for name, build in figures.items():
        fig, caption = build()
        written += save_figure(fig, name, caption, out_dir)
    return written
