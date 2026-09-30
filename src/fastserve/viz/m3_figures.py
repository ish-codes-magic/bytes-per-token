"""M3 figures: quantization from scratch. Each function returns (figure, caption computed from the data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from fastserve.report.m3 import configs, label, single
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, apply_style, save_figure, save_html

Records = list[dict[str, Any]]
# one color and marker per method, everywhere in M3
METHOD_STYLE = {
    "rtn": (BASELINE_GRAY, "o", "RTN"),
    "gptq": (OKABE_ITO["orange"], "^", "GPTQ"),
    "awq": (OKABE_ITO["vermillion"], "v", "AWQ"),
    "rotated": (OKABE_ITO["purple"], "P", "rotation +"),
    "nf4": (OKABE_ITO["green"], "D", "NF4"),
    "fp8_weight": (OKABE_ITO["blue"], "s", "FP8 weights"),
    "library": (OKABE_ITO["black"], "x", "llm-compressor"),
}
GRID_COLORS = {
    "INT4 (textbook)": BASELINE_GRAY,
    "INT4 (full range)": OKABE_ITO["sky_blue"],
    "NF4": OKABE_ITO["green"],
    "FP8 E4M3": OKABE_ITO["blue"],
}


def _style(entry: dict[str, Any]) -> tuple[str, str, str]:
    return METHOD_STYLE["rotated" if entry.get("rotate") else entry["method"]]


def grids(records: Records) -> tuple[plt.Figure, str]:
    """Normalized weights and activations against the points each 4- or 8-bit format can represent."""
    h = single(records, "m3_histograms")
    edges = np.array(h["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    fig = plt.figure(figsize=(11, 4.6))
    spec = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], hspace=0.08, wspace=0.25)
    weights_ax, acts_ax = fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[:, 1])
    grid_ax = fig.add_subplot(spec[1, 0], sharex=weights_ax)

    weights = np.array(h["weights"], dtype=float)
    weights_ax.fill_between(
        centers, weights / weights.sum(), step="mid", color=OKABE_ITO["orange"], alpha=0.6
    )
    weights_ax.set(ylabel="share of weights", title=f"Layer {h['layer']} weights ÷ their group's max")
    weights_ax.tick_params(labelbottom=False)
    names = list(h["grids"])
    for row, name in enumerate(names):
        levels = np.array(h["grids"][name])
        grid_ax.eventplot([levels], lineoffsets=row, linelengths=0.7, colors=GRID_COLORS[name], linewidths=1)
    grid_ax.set(
        yticks=range(len(names)), yticklabels=names, xlabel="value ÷ scale", ylim=(-0.6, len(names) - 0.4)
    )
    grid_ax.grid(False)

    acts = np.array(h["activations"], dtype=float)
    acts_ax.bar(centers, acts / acts.sum(), width=np.diff(edges), color=OKABE_ITO["blue"], log=True)
    acts_ax.set(
        xlabel="activation ÷ its token's max",
        ylabel="share of activations (log)",
        title="q_proj inputs, per token",
    )

    near_zero = np.abs(centers) < 0.25
    share = weights[near_zero].sum() / weights.sum()
    nf4 = sum(abs(v) < 0.25 for v in h["grids"]["NF4"])
    int4 = sum(abs(v) < 0.25 for v in h["grids"]["INT4 (textbook)"])
    caption = (
        f"{share:.0%} of weights lie within ±0.25 of their group's max: NF4 puts {nf4} of its 16 levels "
        f"there, uniform INT4 only {int4}; activations are harsher, with a token's typical value "
        f"{h['activation_abs_max_over_median']:,.0f}× below its max."
    )
    return fig, caption


def _atlas_axis(ax: plt.Axes, matrix: np.ndarray, title: str, norm: LogNorm) -> Any:
    image = ax.imshow(matrix, aspect="auto", cmap="magma", norm=norm, interpolation="nearest")
    ax.set(title=title, xlabel="channel", ylabel="layer")
    ax.grid(False)
    return image


def outlier_atlas(records: Records) -> tuple[plt.Figure, str]:
    """Per-channel activation maxima over (layer × channel): outlier channels stand out as bright columns."""
    a = single(records, "m3_outliers")
    residual, down = np.array(a["residual"]), np.array(a["down_input"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, matrix, title in (
        (left, residual, "Residual stream (each layer's input)"),
        (right, down, "down_proj input (after SwiGLU)"),
    ):
        norm = LogNorm(vmin=max(matrix.min(), 1e-3), vmax=matrix.max())
        fig.colorbar(_atlas_axis(ax, matrix, title, norm), ax=ax, label="max |activation|")
    top = a["top_channels"][:3]
    caption = (
        f"A handful of residual channels ({', '.join(map(str, top))}) reach {a['residual_ratio']:,.0f}× the "
        f"median channel's maximum; down_proj's input peaks at {a['down_input_ratio']:,.0f}× its median."
    )
    return fig, caption


def rotation(records: Records) -> tuple[plt.Figure, str]:
    """The residual stream's channel maxima before and after a random Hadamard rotation (same color scale)."""
    a = single(records, "m3_outliers")
    before, after = np.array(a["residual"]), np.array(a["residual_rotated"])
    norm = LogNorm(vmin=max(min(before.min(), after.min()), 1e-3), vmax=max(before.max(), after.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    image = _atlas_axis(axes[0], before, "Before rotation", norm)
    _atlas_axis(axes[1], after, "After a random Hadamard rotation", norm)
    fig.colorbar(image, ax=axes, label="max |activation|")
    caption = (
        f"Rotating the residual stream spreads its outlier channels over all channels: the largest channel "
        f"maximum drops from {a['residual_ratio']:,.0f}× to {a['residual_rotated_ratio']:,.1f}× the median."
    )
    return fig, caption


def error_vs_bits(records: Records) -> tuple[plt.Figure, str]:
    """KL against storage cost for every weight-only configuration of Qwen3-0.6B."""
    c = configs(records)
    fig, ax = plt.subplots(figsize=(8, 5))
    seen = set()
    for m in c.values():
        e = m["entry"]
        if e["method"] in ("bf16", "w8a8") or m["mean_kl"] <= 0:
            continue
        color, marker, name = _style(e)
        ax.scatter(
            m["bits_per_weight"],
            m["mean_kl"],
            color=color,
            marker=marker,
            s=40,
            label=None if name in seen else name,
            zorder=3,
        )
        seen.add(name)
    for method, prefix in (("rtn", "rtn-int"), ("gptq", "gptq-int"), ("awq", "awq-int")):
        line = sorted(
            (m["bits_per_weight"], m["mean_kl"])
            for n, m in c.items()
            if n.startswith(prefix)
            and "g128" in n
            and m["entry"]["method"] == method
            and not m["entry"].get("rotate")
        )
        if len(line) > 1:
            ax.plot(*zip(*line, strict=True), color=METHOD_STYLE[method][0], alpha=0.6)
    ax.set(
        xlabel="bits per quantized weight (scales included)",
        ylabel="KL vs BF16 (nats/token, log)",
        yscale="log",
    )
    ax.legend(fontsize=8)
    rtn, gptq = c.get("rtn-int4-g128-full"), c.get("gptq-int4-g128")
    caption = "Every bit removed costs roughly an order of magnitude of KL"
    if rtn and gptq:
        caption += (
            f"; at 4 bits GPTQ reaches {gptq['mean_kl'] / rtn['mean_kl']:.0%} of round-to-nearest's KL "
            "on the same grid"
        )
    return fig, caption + "."


def sensitivity_map(records: Records) -> tuple[plt.Figure, str]:
    """KL when a single linear layer is quantized, for every (layer, module type)."""
    scan = single(records, "m3_sensitivity")
    kinds = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layers = 1 + max(cell["layer"] for cell in scan["cells"])
    grid = np.full((len(kinds), layers), np.nan)
    for cell in scan["cells"]:
        grid[kinds.index(cell["module"]), cell["layer"]] = cell["kl"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    positive = grid[grid > 0]
    image = ax.imshow(
        grid,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        norm=LogNorm(vmin=positive.min(), vmax=positive.max()),
    )
    ax.set(
        yticks=range(len(kinds)),
        yticklabels=kinds,
        xlabel="layer",
        title=f"KL when only one module is quantized ({scan['spec']})",
    )
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="KL vs BF16")
    worst = max(scan["cells"], key=lambda cl: cl["kl"])
    by_kind = {k: np.nansum(grid[i]) for i, k in enumerate(kinds)}
    top_kind = max(by_kind, key=by_kind.get)
    caption = (
        f"The single most fragile module is layer {worst['layer']}'s {worst['module']}; summed over layers, "
        f"{top_kind} accounts for {by_kind[top_kind] / np.nansum(grid):.0%} of the damage."
    )
    return fig, caption


def gptq_motion(records: Records) -> tuple[plt.Figure, str]:
    """Snapshots of GPTQ working through a slice of a real layer: rounded columns left, updated ones right."""
    t = single(records, "m3_gptq_trace")
    w, rtn = np.array(t["weights"]), np.array(t["rtn"])
    frames = [np.array(f["weights"]) for f in t["frames"]]
    picks = [0, len(frames) // 4, len(frames) // 2, len(frames) - 1]
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(15, 3.2), sharey=True)
    vmax = np.abs(w).max()
    for ax, i in zip(axes, picks, strict=False):
        ax.imshow(frames[i] - w, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2, interpolation="nearest")
        ax.axvline(i + 0.5, color="black", lw=1)
        ax.set(title=f"after column {i + 1}", xlabel="input column")
        ax.grid(False)
    image = axes[-1].imshow(
        frames[-1] - rtn, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2, interpolation="nearest"
    )
    axes[-1].set(title="GPTQ − RTN (final)", xlabel="input column")
    axes[-1].grid(False)
    axes[0].set_ylabel("output row")
    fig.colorbar(image, ax=axes, label="change in weight")
    caption = (
        f"On a {w.shape[0]}×{w.shape[1]} slice of layer {t['layer']}'s {t['module']} ({t['spec']}), each "
        f"rounded column's error is pushed onto the columns to its right; the output error ends "
        f"{t['loss_rtn'] / t['loss_gptq']:.1f}× lower than round-to-nearest's."
    )
    return fig, caption


def gptq_animation(records: Records) -> Any:
    """The same trace as an interactive Plotly animation: one frame per rounded column."""
    import plotly.graph_objects as go

    t = single(records, "m3_gptq_trace")
    w = np.array(t["weights"])
    vmax = float(np.abs(w).max() / 2)
    frames = [
        go.Frame(
            data=[go.Heatmap(z=np.array(f["weights"]) - w, zmin=-vmax, zmax=vmax, colorscale="RdBu_r")],
            name=str(i),
            layout=go.Layout(
                title=f"GPTQ: column {i + 1} rounded; its error spread to the columns on its right"
            ),
        )
        for i, f in enumerate(t["frames"])
    ]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="GPTQ in motion: change of each weight from its original value",
        xaxis_title="input column",
        yaxis_title="output row",
        updatemenus=[
            {"type": "buttons", "buttons": [{"label": "Play", "method": "animate", "args": [None]}]}
        ],
        sliders=[
            {
                "steps": [
                    {"label": str(i + 1), "method": "animate", "args": [[str(i)]]} for i in range(len(frames))
                ]
            }
        ],
    )
    return fig


def pareto(records: Records) -> tuple[plt.Figure, str]:
    """Quality against model size for every Qwen3-0.6B configuration, with the frontier drawn in."""
    c = configs(records)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    points = []
    seen = set()
    for m in c.values():
        e = m["entry"]
        if m["mean_kl"] <= 0 or e["method"] == "w8a8":
            continue
        color, marker, name = _style(e)
        ax.scatter(
            m["model_gb"],
            m["mean_kl"],
            color=color,
            marker=marker,
            s=40,
            zorder=3,
            label=None if name in seen else name,
        )
        seen.add(name)
        points.append((m["model_gb"], m["mean_kl"], label(e)))
    frontier, best = [], float("inf")
    for gb, kl, name in sorted(points):
        if kl < best:
            frontier.append((gb, kl, name))
            best = kl
    ax.step([p[0] for p in frontier], [p[1] for p in frontier], where="post", color=BASELINE_GRAY, alpha=0.5)
    if "bf16" in c:
        ax.axvline(c["bf16"]["model_gb"], color=BASELINE_GRAY, ls="--", label="BF16 size")
    ax.set(xlabel="model size (GB, embeddings in BF16)", ylabel="KL vs BF16 (log)", yscale="log")
    ax.legend(fontsize=8)
    smallest = min(frontier, key=lambda p: p[0]) if frontier else None
    caption = (
        f"The frontier's smallest point is {smallest[2]} at {smallest[0]:.2f} GB; below that, the BF16 "
        "embedding (a quarter of the weights) sets a floor on size that no weight quantizer moves."
        if smallest
        else "No configurations."
    )
    return fig, caption


FIGURES = {
    "m3_grids": grids,
    "m3_outlier_atlas": outlier_atlas,
    "m3_rotation": rotation,
    "m3_error_vs_bits": error_vs_bits,
    "m3_sensitivity": sensitivity_map,
    "m3_gptq_motion": gptq_motion,
    "m3_pareto": pareto,
}


def make_all(records: Records, out_dir: str | Path) -> list[Path]:
    apply_style()
    written: list[Path] = []
    for name, build in FIGURES.items():
        fig, caption = build(records)
        written += save_figure(fig, name, caption, out_dir)
    written.append(save_html(gptq_animation(records), "m3_gptq_motion", out_dir))
    return written
