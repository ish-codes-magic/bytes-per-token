"""M0 figures: what the GPU really does, measured against what the datasheet promises.

Every function takes the records of one probe run and returns (figure, caption). The caption states the
takeaway using numbers computed from the data, never typed by hand.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fastserve.hw.analysis import gpu_spec, measured_bandwidth, measured_peak_flops, metrics_of
from fastserve.perfmodel.roofline import matmul_intensity, ridge_point
from fastserve.viz.style import BASELINE_GRAY, apply_style, save_figure, series_style

Records = list[dict[str, Any]]
FORMAT_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}


def _size_label(n: float) -> str:
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= scale:
            return f"{n / scale:g} {unit}"
    return f"{n:g} B"


def bandwidth_vs_size(records: Records) -> tuple[plt.Figure, str]:
    spec = gpu_spec(records)
    rows = metrics_of(records, "bandwidth")
    fig, ax = plt.subplots()
    for method, label in (("read", "read (Triton kernel)"), ("copy", "copy (read + write)")):
        points = sorted((m["size_bytes"], m["gbps_median"]) for m in rows if m["method"] == method)
        ax.plot(*zip(*points, strict=True), label=label, **series_style(method))

    sizes = sorted({m["size_bytes"] for m in rows})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [_size_label(s) for s in sizes], rotation=45)
    ax.set(
        xlabel="transfer size", ylabel="achieved bandwidth (GB/s, median)", title="Memory bandwidth vs size"
    )

    measured = measured_bandwidth(records)
    caption = f"Large reads reach {measured / 1e9:.0f} GB/s." if measured else "No large-transfer data."
    if spec:
        ax.axhline(spec["bandwidth"] / 1e9, ls="--", color=BASELINE_GRAY, label="datasheet bandwidth")
        ax.axvline(
            spec["l2_bytes"], ls=":", color=BASELINE_GRAY, label=f"L2 cache ({_size_label(spec['l2_bytes'])})"
        )
        in_cache = [m["gbps_median"] for m in rows if m["size_bytes"] < spec["l2_bytes"]]
        if measured:
            caption = (
                f"Large reads reach {measured / 1e9:.0f} GB/s, {measured / spec['bandwidth']:.0%} of the "
                f"{spec['bandwidth'] / 1e9:.0f} GB/s datasheet figure"
            )
            if in_cache and max(in_cache) * 1e9 > 1.1 * measured:
                caption += f"; transfers that fit in L2 peak at {max(in_cache):.0f} GB/s (cache, not memory)"
            caption += "."
    ax.legend()
    return fig, caption


def roofline(records: Records) -> tuple[plt.Figure, str]:
    spec = gpu_spec(records)
    bandwidth = measured_bandwidth(records)
    peaks = measured_peak_flops(records)
    fig, ax = plt.subplots()
    intensity = np.logspace(-1, 4, 300)  # FLOPs per byte

    ridges: dict[str, float] = {}
    for fmt in ("bf16", "fp8", "int8"):
        style = series_style(fmt)
        if bandwidth and fmt in peaks:
            roof = np.minimum(peaks[fmt], bandwidth * intensity) / 1e12
            ax.plot(intensity, roof, color=style["color"], label=f"measured {fmt.upper()}")
            ridges[fmt] = ridge_point(peak_flops=peaks[fmt], bandwidth=bandwidth)
            ax.plot(ridges[fmt], peaks[fmt] / 1e12, ls="none", ms=8, **style)
        if spec and fmt in spec["peak_flops"]:
            roof = np.minimum(spec["peak_flops"][fmt], spec["bandwidth"] * intensity) / 1e12
            ax.plot(
                intensity,
                roof,
                ls="--",
                lw=1,
                color=style["color"],
                alpha=0.6,
                label=f"datasheet {fmt.upper()}",
            )

    # Every measured BF16 matmul at its arithmetic intensity: small M (decode-like) sits under the slope.
    points = [
        (
            matmul_intensity(m["m"], m["n"], m["k"], a_bytes=2, b_bytes=2, out_bytes=2),
            m["tflops_median"],
            m["m"],
        )
        for m in metrics_of(records, "matmul")
        if m["format"] == "bf16" and "tflops_median" in m
    ]
    if points:
        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            s=14,
            color=BASELINE_GRAY,
            marker="x",
            alpha=0.8,
            label="BF16 matmuls, cold L2 (one per shape)",
        )

    ax.set(
        xscale="log",
        yscale="log",
        xlabel="arithmetic intensity (FLOPs/byte)",
        ylabel="TFLOP/s",
        title="Roofline: measured vs datasheet",
    )
    ax.legend(fontsize=8)

    caption = "Not enough data for a roofline."
    if "bf16" in ridges:
        caption = f"Measured BF16 ridge point: {ridges['bf16']:.0f} FLOPs/byte"
        if "fp8" in ridges:
            caption += f" (FP8: {ridges['fp8']:.0f})"
        batch1 = [p for p in points if p[2] == 1]
        if batch1:
            ai, tflops, _ = min(batch1, key=lambda p: p[0])
            caption += (
                f"; an M=1 matmul (decode at batch 1) sits at {ai:.1f} FLOPs/byte and reaches "
                f"{tflops * 1e12 / peaks['bf16']:.1%} of the measured peak"
            )
        caption += "."
    return fig, caption


def matmul_efficiency(records: Records) -> tuple[plt.Figure, str]:
    spec = gpu_spec(records)
    rows = metrics_of(records, "matmul")
    ms = sorted({m["m"] for m in rows})
    nks = sorted({m["n"] for m in rows})
    formats = [
        f
        for f in ("bf16", "fp8", "int8")
        if spec and f in spec["peak_flops"] and any(m["format"] == f for m in rows)
    ]

    fig, axes = plt.subplots(1, max(len(formats), 1), figsize=(4 * max(len(formats), 1), 4), squeeze=False)
    image = None
    for ax, fmt in zip(axes[0], formats, strict=False):
        grid = np.full((len(ms), len(nks)), np.nan)
        for m in rows:
            if m["format"] == fmt and "tflops_median" in m:
                grid[ms.index(m["m"]), nks.index(m["n"])] = (
                    100 * m["tflops_median"] * 1e12 / spec["peak_flops"][fmt]
                )
        image = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=100, aspect="auto")
        for i in range(len(ms)):
            for j in range(len(nks)):
                value = grid[i, j]
                text = "n/a" if math.isnan(value) else f"{value:.0f}" if value >= 1 else f"{value:.1f}"
                color = "black" if not math.isnan(value) and value > 60 else "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(len(nks)), [str(n) for n in nks])
        ax.set_yticks(range(len(ms)), [str(m) for m in ms])
        ax.set(xlabel="N = K (layer width)", ylabel="M (≈ batch size)", title=fmt.upper())
        ax.grid(False)
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), label="% of datasheet peak (n/a = unsupported)")
    fig.suptitle("Matmul efficiency", fontweight="bold")

    caption = "No matmul data."
    bf16 = [m for m in rows if m["format"] == "bf16" and "tflops_median" in m]
    if spec and bf16 and "bf16" in spec["peak_flops"]:
        peak = spec["peak_flops"]["bf16"]
        best = max(bf16, key=lambda m: m["tflops_median"])
        smallest_m = [m for m in bf16 if m["m"] == min(ms)]
        best_pct = 100 * best["tflops_median"] * 1e12 / peak
        caption = f"BF16 reaches {best_pct:.0f}% of datasheet peak at M={best['m']}"
        if smallest_m:
            low = min(smallest_m, key=lambda m: m["tflops_median"])
            caption += (
                f" but only {100 * low['tflops_median'] * 1e12 / peak:.1f}% at M={low['m']}: "
                "decode-sized matmuls can't fill the GPU"
            )
        caption += "."
    return fig, caption


FIGURES = {
    "hw_bandwidth_vs_size": bandwidth_vs_size,
    "hw_roofline": roofline,
    "hw_matmul_efficiency": matmul_efficiency,
}


def make_all(records: Records, out_dir: str | Path) -> list[Path]:
    """Render every M0 figure for one probe run into out_dir. Returns all files written."""
    apply_style()
    written: list[Path] = []
    for name, build in FIGURES.items():
        fig, caption = build(records)
        written += save_figure(fig, name, caption, out_dir)
    return written
