"""M7 figures: the custom kernels. Each returns (figure, caption computed from the data)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, Rectangle

from fastserve.report.m7 import (
    _one,
    _rows,
    att,
    attention_cases,
    nanoserve_cases,
    norm_row,
    norm_sizes,
    step,
    tune_best,
    tune_cells,
)
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, SERIES, apply_style, save_figure

Records = list[dict[str, Any]]
LONG = (1, 32768)
# One color per number format, as everywhere; a solid line is one of our kernels, a dashed one is not.
ATT_STYLE = {
    "sdpa-bf16": ("PyTorch attention, BF16", SERIES["bf16"], "--"),
    "flashinfer-fp16": ("FlashInfer, FP16", SERIES["fp16"], "--"),
    "triton-bf16": ("Kernel 2, BF16", SERIES["bf16"], "-"),
    "triton-int8": ("Kernel 2, INT8", SERIES["int8"], "-"),
    "triton-int4": ("Kernel 2, INT4", SERIES["int4"], "-"),
}
NQ_STYLE = {
    "vllm-separate": ("vLLM, two ops (INT8)", SERIES["bf16"], "--"),
    "vllm-fused-fp8": ("vLLM fused norm + FP8", SERIES["fp8"], "--"),
    "triton-fused": ("Kernel 1 (INT8)", SERIES["int8"], "-"),
}
CACHE_SHORT = {
    "bf16-sdpa": "BF16\nPyTorch attention",
    "bf16-kernel": "BF16\nkernel 2",
    "int8-kernel": "INT8 codes\nkernel 2",
    "int4-kernel": "INT4 codes\nkernel 2",
}


def _shape(batch: int, context: int) -> str:
    return f"{batch} × {context:,}"


# ---- 1. roofline -------------------------------------------------------------------------------------------


def kernel_roofline(m7: Records, bandwidth: float) -> tuple[plt.Figure, str]:
    """Bytes ÷ time for every contender, against the bandwidth M0 measured."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ceiling = bandwidth / 1e9

    cases = [m for m in attention_cases(m7) if m["batch"] == 1]
    for name, (label, (color, marker), line) in ATT_STYLE.items():
        points = [
            (m["contenders"][name]["cache_bytes"], m["contenders"][name]["gbps"])
            for m in cases
            if "gbps" in m["contenders"].get(name, {})
        ]
        if points:
            xs, ys = zip(*sorted(points), strict=True)
            left.plot(xs, ys, line, color=color, marker=marker, label=label)
    left.axhline(ceiling, color=OKABE_ITO["vermillion"], linestyle=":", label="M0: measured read bandwidth")
    left.set(
        xscale="log",
        xlabel="KV cache bytes one layer's attention must read (one sequence, 512 → 32,768 tokens)",
        ylabel="cache bytes ÷ time (GB/s)",
        title="Kernel 2: decode attention",
    )
    left.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))

    d = max(d for d, _ in norm_sizes(m7, "norm_quant"))
    for name, (label, (color, marker), line) in NQ_STYLE.items():
        points = []
        for width, rows in norm_sizes(m7, "norm_quant"):
            entry = (norm_row(m7, rows, width, "norm_quant") or {}).get("contenders", {}).get(name, {})
            ms = (entry.get("graph") or {}).get("ms")
            if width == d and ms and entry.get("bytes_moved"):
                points.append((entry["bytes_moved"], entry["bytes_moved"] / (ms / 1e3) / 1e9))
        if points:
            xs, ys = zip(*sorted(points), strict=True)
            right.plot(xs, ys, line, color=color, marker=marker, label=label)
    right.axhline(ceiling, color=OKABE_ITO["vermillion"], linestyle=":", label="M0: measured read bandwidth")
    right.set(
        xscale="log",
        yscale="log",
        xlabel=f"bytes moved (1 → 32,768 tokens × {d:,}), replayed from a CUDA graph",
        ylabel="bytes moved ÷ time (GB/s)",
        title="Kernel 1: RMSNorm + quantization",
    )
    right.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    def share(name: str) -> float:
        return att(m7, *LONG, name, "gbps") / ceiling

    big = norm_row(m7, 32768, d, "norm_quant")["contenders"]["triton-fused"]
    k1 = big["bytes_moved"] / (big["graph"]["ms"] / 1e3) / bandwidth
    caption = (
        f"At 1 × 32,768 tokens kernel 2 reads the BF16 cache at {share('triton-bf16'):.0%} of the measured "
        f"bandwidth and the INT4 cache at {share('triton-int4'):.0%}, where PyTorch's path manages "
        f"{share('sdpa-bf16'):.0%}; kernel 1 moves its bytes at {k1:.0%} once they no longer fit in the "
        "cache (above the line, the data is in L2)."
    )
    return fig, caption


# ---- 2. speedup heatmaps -----------------------------------------------------------------------------------


def _ratio_grid(m7: Records, slow: str, fast: str) -> tuple[np.ndarray, list[int], list[int]]:
    cases = attention_cases(m7)
    batches = sorted({m["batch"] for m in cases})
    contexts = sorted({m["context"] for m in cases})
    grid = np.full((len(batches), len(contexts)), np.nan)
    for m in cases:
        a, b = (m["contenders"].get(n, {}).get("ms") for n in (slow, fast))
        if a and b:
            grid[batches.index(m["batch"]), contexts.index(m["context"])] = a / b
    return grid, batches, contexts


def speedup_heatmaps(m7: Records) -> tuple[plt.Figure, str]:
    """Kernel 2 over (batch, context): against PyTorch, against FlashInfer, and bytes vs fusion apart."""
    panels = [
        ("INT4 kernel\nvs PyTorch attention (BF16)", "sdpa-bf16", "triton-int4"),
        ("Fusion alone: BF16 kernel\nvs PyTorch attention (BF16)", "sdpa-bf16", "triton-bf16"),
        ("Bytes alone: INT4 codes\nvs BF16, same kernel", "triton-bf16", "triton-int4"),
        ("INT4 kernel\nvs FlashInfer (FP16), one request", "flashinfer-fp16", "triton-int4"),
    ]
    grids = [_ratio_grid(m7, slow, fast) for _, slow, fast in panels]
    top = max(float(np.nanmax(g)) for g, _, _ in grids)
    low = min(float(np.nanmin(g)) for g, _, _ in grids)
    span = max(top, 1 / low, 1.5)
    norm = LogNorm(vmin=1 / span, vmax=span)  # 1× in the middle: purple loses, orange wins
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), layout="constrained")
    for ax, (title, _, _), (grid, batches, contexts) in zip(axes, panels, grids, strict=True):
        image = ax.imshow(grid, cmap="PuOr_r", norm=norm, aspect="auto")
        for i in range(len(batches)):
            for j in range(len(contexts)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", color=BASELINE_GRAY)
                    continue
                dark = abs(np.log(grid[i, j])) > 0.6 * np.log(span)
                text = f"{grid[i, j]:.2f}×" if grid[i, j] < 10 else f"{grid[i, j]:.0f}×"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color="white" if dark else "black")
        ax.set_xticks(range(len(contexts)), [f"{c:,}" for c in contexts])
        ax.set_yticks(range(len(batches)), [str(b) for b in batches])
        ax.set(xlabel="context (tokens)", ylabel="batch (sequences)")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
    fig.colorbar(image, ax=axes, label="time of the other ÷ time of kernel 2", shrink=0.9)

    vs_torch, vs_fi = grids[0][0], grids[3][0]
    losses = int(np.sum(vs_torch < 1))
    caption = (
        f"Kernel 2 on INT4 codes is {np.nanmin(vs_torch):.1f}–{np.nanmax(vs_torch):.0f}× the speed of "
        f"nanoserve's PyTorch attention ({losses} of {int(np.sum(~np.isnan(vs_torch)))} shapes slower), and "
        f"{np.nanmin(vs_fi):.2f}–{np.nanmax(vs_fi):.2f}× FlashInfer's on a full-precision cache: "
        "fewer bytes win at long context, launch overhead decides the short ones."
    )
    return fig, caption


# ---- 3. autotuning landscape -------------------------------------------------------------------------------


def tuning_landscape(m7: Records) -> tuple[plt.Figure, str]:
    """Kernel 2's runtime over (tokens per program × warps), relative to the best cell of each panel."""
    shapes = sorted({(m["batch"], m["context"], m["bits"]) for m in _rows(m7, "m7_tune")})
    wanted = [s for s in ((1, 32768, 4), (64, 8192, 4), (1, 32768, 16)) if s in shapes] or shapes[:3]
    fig, axes = plt.subplots(1, len(wanted), figsize=(5.2 * len(wanted), 4.6), squeeze=False)
    for ax, (batch, context, bits) in zip(axes[0], wanted, strict=True):
        cells = tune_cells(m7, batch, context, bits)
        splits = sorted({m["split"] for m in cells})
        warps = sorted({m["num_warps"] for m in cells})
        best = min(m["ms"] for m in cells)
        grid = np.full((len(splits), len(warps)), np.nan)
        for m in cells:
            grid[splits.index(m["split"]), warps.index(m["num_warps"])] = m["ms"] / best
        image = ax.imshow(
            grid, cmap="viridis_r", norm=LogNorm(vmin=1, vmax=max(2.0, float(np.nanmax(grid)))), aspect="auto"
        )
        for i in range(len(splits)):
            for j in range(len(warps)):
                if not np.isnan(grid[i, j]):
                    color = "white" if grid[i, j] > np.sqrt(np.nanmax(grid)) else "black"
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(len(warps)), [str(w) for w in warps])
        labels = [f"{s:,}" + (" (no split)" if s == context else "") for s in splits]
        ax.set_yticks(range(len(splits)), labels)
        fmt = "BF16" if bits == 16 else f"INT{bits}"
        ax.set(xlabel="warps per program", title=f"{fmt}, {_shape(batch, context)}")
        if ax is axes[0][0]:
            ax.set_ylabel("tokens per program")
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="time ÷ best")
    fig.tight_layout()

    best = tune_best(m7, *LONG, 4)
    cells = tune_cells(m7, *LONG, 4)
    whole = min(m["ms"] for m in cells if m["split"] == LONG[1])
    same_split = {m["num_warps"]: m["ms"] for m in cells if m["split"] == best["split"]}
    worst = max(same_split, key=same_split.get)  # the costliest warp count at the best split
    plural = "" if best["num_warps"] == 1 else "s"
    caption = (
        f"For INT4 at 1 × 32,768 the best cell is {best['split']} tokens per program with "
        f"{best['num_warps']} warp{plural} ({best['programs']:,} programs): not splitting is "
        f"{whole / best['ms']:.0f}× slower, and the wrong warp count at that split ({worst}) "
        f"{same_split[worst] / best['ms']:.1f}×."
    )
    return fig, caption


# ---- 4. one decode step, before and after ------------------------------------------------------------------

_TIMELINE_KINDS = [  # the first pattern that matches a kernel's name decides its kind
    ("kernel 2", re.compile(r"decode_attention", re.I), OKABE_ITO["orange"]),
    ("PyTorch's attention kernel", re.compile(r"fmha|attention", re.I), OKABE_ITO["vermillion"]),
    ("matmul (the weights)", re.compile(r"gemm|gemv|cutlass|xmma|cublas|splitk", re.I), OKABE_ITO["blue"]),
    ("indexing (reading the cache out)", re.compile(r"index|gather|scatter", re.I), OKABE_ITO["purple"]),
    ("elementwise (copies, masks, norms)", re.compile(r"elementwise", re.I), OKABE_ITO["sky_blue"]),
]
_OTHER = ("everything else", BASELINE_GRAY)


def _kind(name: str) -> tuple[str, str]:
    return next(((label, color) for label, pattern, color in _TIMELINE_KINDS if pattern.search(name)), _OTHER)


_ATTENTION_KINDS = ("kernel 2", "PyTorch's attention kernel")


def _draw_lane(ax, trace: dict[str, Any], window: tuple[float, float] | None = None) -> dict[str, float]:
    """One trace as colored bars on a time axis (ms). Returns GPU-busy ms per kind of kernel."""
    kinds = [_kind(name) for name in trace["names"]]
    order = [kind[0] for kind in (*_TIMELINE_KINDS, _OTHER)]
    busy: dict[str, float] = {}
    for label, color in sorted(set(kinds), key=lambda kind: order.index(kind[0])):
        bars = [(start / 1e3, dur / 1e3) for i, start, dur in trace["events"] if kinds[i][0] == label]
        busy[label] = sum(dur for _, dur in bars)
        # No edge: a 5 µs kernel must not be drawn a pixel wide, or a mostly idle GPU looks busy.
        ax.broken_barh(
            bars, (0, 1), facecolors=color, edgecolor="none", label=f"{label}: {busy[label]:.1f} ms"
        )
    if window is not None:
        ax.set_xlim(*window)
    ax.set(yticks=[], ylim=(0, 1))
    ax.grid(False)
    return busy


def _one_layer(trace: dict[str, Any]) -> tuple[float, float] | None:
    """The stretch between two consecutive attention kernels in the middle of the step: one layer (ms)."""
    kinds = [_kind(name)[0] for name in trace["names"]]
    starts = [start / 1e3 for i, start, _ in trace["events"] if kinds[i] in _ATTENTION_KINDS]
    middle = len(starts) // 2
    return (starts[middle], starts[middle + 1]) if len(starts) > middle + 1 else None


def step_timeline(m7: Records) -> tuple[plt.Figure, str]:
    """Every GPU kernel of one nanoserve decode step, before and after: the whole step, and one layer of it.

    The traces come from PyTorch's profiler, which slows every launch: the steps are longer here than in
    the timing tables, but each kernel's own duration is the GPU's.
    """
    traces = [_one(m7, "m7_timeline", cache=cache) for cache in ("bf16-sdpa", "int4-kernel")]
    titles = ("before: BF16 cache, PyTorch attention", "after: INT4 codes, kernel 2")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [2.2, 1]})
    totals = {}
    for (whole, zoom), trace, title in zip(axes, traces, titles, strict=True):
        busy = _draw_lane(whole, trace)
        end = max(start + dur for _, start, dur in trace["events"]) / 1e3
        totals[trace["cache"]] = (end, sum(busy.values()), len(trace["events"]))
        whole.set_title(
            f"{title}: {len(trace['events']):,} kernels, GPU busy {sum(busy.values()):.1f} of {end:.1f} ms",
            fontsize=10,
            loc="left",
        )
        below = -0.22 if whole is axes[0][0] else -0.42  # the bottom row's legend clears its x label
        whole.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, below))
        window = _one_layer(trace)
        _draw_lane(zoom, trace, window)
        layer = f"one layer ({window[1] - window[0]:.2f} ms)" if window else "the same step"
        zoom.set_title(f"zoom: {layer}", fontsize=10, loc="left")
    shape = _shape(traces[0]["batch"], traces[0]["context"])
    for ax in axes[1]:
        ax.set_xlabel("time under the profiler (ms)")
    fig.suptitle(
        f"One nanoserve decode step, {shape} tokens: every GPU kernel", fontsize=11, x=0.01, ha="left"
    )
    fig.tight_layout()
    (b_end, b_busy, _), (a_end, a_busy, a_n) = totals["bf16-sdpa"], totals["int4-kernel"]
    caption = (
        f"One decode step at {shape} tokens: the GPU works {b_busy:.0f} of {b_end:.0f} ms before and "
        f"{a_busy:.0f} of {a_end:.0f} ms after; what is left is {a_n:,} small kernels with gaps between "
        "them (Python between launches), which no attention kernel can shorten."
    )
    return fig, caption


# ---- 5. memory traffic -------------------------------------------------------------------------------------


def _lane(ax, y: float, title: str, flows: list[tuple[str, str, float]], total: str) -> None:
    """One pipeline: a row of kernels above a strip of memory, with an arrow per read or write."""
    ax.add_patch(Rectangle((0, y), 10, 0.5, color="#E8E8E8"))
    ax.text(0.1, y + 0.25, "memory (HBM)", va="center", fontsize=8, color=BASELINE_GRAY)
    ax.text(0, y + 2.75, title, fontsize=10, fontweight="bold")
    ax.text(10, y + 2.75, total, fontsize=9, ha="right")
    kernels = list(dict.fromkeys(kernel for kernel, _, _ in flows))
    width = 8.0 / len(kernels)
    for i, kernel in enumerate(kernels):
        x = 1.6 + i * width
        ax.add_patch(Rectangle((x, y + 1.6), width * 0.85, 0.8, color=OKABE_ITO["sky_blue"], alpha=0.6))
        ax.text(x + width * 0.425, y + 2.0, kernel, ha="center", va="center", fontsize=8)
        mine = [(direction, nbytes) for k, direction, nbytes in flows if k == kernel]
        for j, (direction, nbytes) in enumerate(mine):
            px = x + width * 0.85 * (j + 1) / (len(mine) + 1)
            up = direction == "read"
            arrow = FancyArrowPatch(
                (px, y + 0.5) if up else (px, y + 1.6),
                (px, y + 1.6) if up else (px, y + 0.5),
                arrowstyle="-|>",
                mutation_scale=12,
                color=OKABE_ITO["blue"] if up else OKABE_ITO["vermillion"],
                linewidth=1 + 3 * nbytes / max(n for _, _, n in flows),
            )
            ax.add_patch(arrow)
            ax.text(px + 0.08, y + 1.05, f"{direction}\n{nbytes:,.0f} B", fontsize=7.5, va="center")


def memory_traffic(cfg: Any, group: int = 32) -> tuple[plt.Figure, str]:
    """Bytes per token between memory and the kernels, unfused and fused, from the model's own sizes."""
    d, hd = cfg.hidden_size, cfg.head_dim
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax in (left, right):
        ax.set(xlim=(0, 10), ylim=(0, 6.6))
        ax.axis("off")

    norm = [("rms_norm", "read", 2 * d), ("rms_norm", "write", 2 * d)]
    quant = [("scaled_int8_quant", "read", 2 * d), ("scaled_int8_quant", "write", d + 4)]
    fused1 = [("kernel 1", "read", 2 * d), ("kernel 1", "write", d + 4)]
    unfused1_total, fused1_total = sum(n for _, _, n in norm + quant), sum(n for _, _, n in fused1)
    left.set_title(f"Kernel 1: one token's hidden state (d = {d:,})", fontsize=11)
    _lane(left, 3.4, "two ops", norm + quant, f"{unfused1_total:,} bytes")
    _lane(left, 0.0, "fused", fused1, f"{fused1_total:,} bytes")

    key_grid = 2 * 2 * hd // group  # a float16 scale and zero-point per channel, shared by `group` tokens
    value_grid = 2 * 2  # a float16 scale and zero-point per token
    codes = hd // 2 + hd // 2 + key_grid + value_grid  # 4-bit K and V: two channels per byte
    floats = 2 * hd * 2  # K and V in BF16
    unfused2 = [("dequantize", "read", codes), ("dequantize", "write", floats), ("attention", "read", floats)]
    fused2 = [("kernel 2", "read", codes)]
    unfused2_total = sum(n for _, _, n in unfused2)
    right.set_title(f"Kernel 2: one cached token of one KV head (D = {hd}), INT4", fontsize=11)
    _lane(right, 3.4, "dequantize, then attend", unfused2, f"{unfused2_total:,} bytes")
    _lane(right, 0.0, "fused", fused2, f"{codes:,} bytes")
    fig.tight_layout()
    caption = (
        f"Fusing moves {unfused1_total / fused1_total:.1f}× fewer bytes per token in kernel 1 "
        f"({unfused1_total:,} → {fused1_total:,}) and {unfused2_total / codes:.1f}× fewer per cached token "
        f"in kernel 2 ({unfused2_total:,} → {codes}); a BF16 cache costs {floats} bytes for the same token."
    )
    return fig, caption


# ---- 6. waterfall v4 ---------------------------------------------------------------------------------------


def waterfall_v4(m7: Records, dollars_per_hour: float) -> tuple[plt.Figure, str]:
    """nanoserve's $ per 1M tokens through each cache, at the longest context and at a batch."""
    cases = [c for c in nanoserve_cases(m7) if step(m7, "int4-kernel", *c)]
    cases = sorted(cases, key=lambda c: c[0] * c[1])[-2:]  # the two largest caches
    fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 4.6), squeeze=False)
    found = {}
    for ax, (batch, context) in zip(axes[0], cases, strict=True):
        bars = []
        for cache in CACHE_SHORT:
            m = step(m7, cache, batch, context)
            if m:
                bars.append((cache, dollars_per_hour / (m["tokens_per_s"] * 3600) * 1e6))
        base = bars[0][1]
        for i, (_cache, cost) in enumerate(bars):
            if i == 0:
                ax.bar(i, cost, color=BASELINE_GRAY)
                continue
            ax.bar(
                i,
                cost - base,
                bottom=base,
                color=OKABE_ITO["green"] if cost < base else OKABE_ITO["vermillion"],
            )
            ax.annotate(f"{cost / base - 1:+.0%}", (i, max(cost, base)), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(bars)), [CACHE_SHORT[cache] for cache, _ in bars], fontsize=8)
        ax.set(
            title=f"nanoserve, Qwen3-0.6B, {_shape(batch, context)} tokens", ylabel="$ per 1M output tokens"
        )
        ax.set_ylim(0, max(cost for _, cost in bars) * 1.15)
        ax.set_axisbelow(True)
        found[(batch, context)] = {cache: cost / base - 1 for cache, cost in bars[1:]}
    fig.tight_layout()
    batch, context = max(cases, key=lambda c: c[1])  # the caption is about the longest context
    changes = found[(batch, context)]
    kl = _one(m7, "m7_kl", cache="int4-kernel")
    quality = f" (KL {kl['mean_kl']:.2g} from the BF16 cache)" if kl else ""
    smaller = (
        step(m7, "bf16-kernel", batch, context)["kv_bytes"]
        / step(m7, "int4-kernel", batch, context)["kv_bytes"]
    )
    caption = (
        f"Waterfall v4, nanoserve at {_shape(batch, context)} tokens: reading the same BF16 cache with "
        f"kernel 2 changes cost by {changes['bf16-kernel']:+.0%}, and INT4 codes by "
        f"{changes['int4-kernel']:+.0%}{quality}: once attention is fused the step is bound by Python's "
        f"launches, so INT4 buys a {smaller:.1f}× smaller cache, not time. These bars are nanoserve's, "
        "not vLLM's."
    )
    return fig, caption


def make_all(
    m7: Records, bandwidth: float, cfg: Any, dollars_per_hour: float, out_dir: str | Path
) -> list[Path]:
    apply_style()
    figures = {
        "m7_roofline": lambda: kernel_roofline(m7, bandwidth),
        "m7_speedup": lambda: speedup_heatmaps(m7),
        "m7_tuning": lambda: tuning_landscape(m7),
        "m7_timeline": lambda: step_timeline(m7),
        "m7_traffic": lambda: memory_traffic(cfg),
        "m7_waterfall": lambda: waterfall_v4(m7, dollars_per_hour),
    }
    written: list[Path] = []
    for name, build in figures.items():
        try:
            fig, caption = build()
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as err:  # data for this figure not in yet
            print(f"skipping {name}: {type(err).__name__}: {err}")
            continue
        written += save_figure(fig, name, caption, out_dir)
    return written
