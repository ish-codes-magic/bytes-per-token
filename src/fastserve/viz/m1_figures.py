"""M1 figures: nanoserve on the real Qwen3-0.6B. Each returns (figure, caption computed from data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fastserve.engine.config import ModelConfig
from fastserve.hw.analysis import measured_bandwidth, measured_peak_flops, metrics_of
from fastserve.report.m1 import batching_summary, component_summary, decode_flops_bytes, prefill_flops_bytes
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, apply_style, save_figure, save_html, series_style

Records = list[dict[str, Any]]

COMPONENT_COLORS = {
    "embedding": OKABE_ITO["yellow"],
    "RMSNorm": OKABE_ITO["sky_blue"],
    "attention block": OKABE_ITO["blue"],
    "MLP": OKABE_ITO["orange"],
    "LM head": OKABE_ITO["vermillion"],
    "other (RoPE tables, sampling, Python)": BASELINE_GRAY,
}


def decode_scaling(m1: Records) -> tuple[plt.Figure, str]:
    rows = sorted(metrics_of(m1, "decode_speed"), key=lambda m: m["batch"])
    batch = np.array([m["batch"] for m in rows])
    tok_s = np.array([m["tokens_per_s"] for m in rows])
    step = np.array([m["step_ms_median"] for m in rows])
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(batch, tok_s, label="nanoserve (measured)", **series_style("baseline"))
    left.plot(batch, tok_s[0] * batch / batch[0], ls="--", color=BASELINE_GRAY, lw=1, label="perfect scaling")
    left.set(xscale="log", yscale="log", xlabel="batch size", ylabel="decode tokens/s", title="Throughput")
    left.legend()
    right.plot(batch, step, **series_style("baseline"))
    right.set(
        xscale="log", xlabel="batch size", ylabel="ms per decode step", title="Step time", ylim=(0, None)
    )
    for ax in (left, right):
        ax.set_xticks(batch, [str(b) for b in batch])
    fig.suptitle("nanoserve decode: batching is nearly free", fontweight="bold")
    i64 = int(np.argmin(np.abs(batch - 64)))
    caption = (
        f"Batch 1 decodes {tok_s[0]:.0f} tokens/s; batch {batch[i64]} reaches {tok_s[i64]:,.0f} "
        f"({tok_s[i64] / tok_s[0]:.0f}×) while each step takes only {step[i64] / step[0]:.1f}× longer."
    )
    return fig, caption


def anatomy(m1: Records) -> tuple[plt.Figure, str]:
    parts = component_summary(m1)
    labels = list(parts)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for row, label in enumerate(labels):
        total = sum(parts[label].values())
        left = 0.0
        for name, color in COMPONENT_COLORS.items():
            ms = parts[label].get(name, 0.0)
            share = 100 * ms / total
            ax.barh(row, share, left=left, color=color, edgecolor="white", label=name if row == 0 else None)
            if share >= 6:
                ax.text(
                    left + share / 2, row, f"{ms:.1f} ms", ha="center", va="center", fontsize=8, color="white"
                )
            left += share
    ax.set_yticks(range(len(labels)), [f"{lbl}\n({sum(parts[lbl].values()):.1f} ms)" for lbl in labels])
    ax.set(
        xlabel="% of step time (GPU timeline, launch gaps included)",
        xlim=(0, 100),
        title="Where a step's time goes",
    )
    ax.legend(ncols=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.grid(False)
    decode = next((lbl for lbl in labels if lbl.startswith("decode")), labels[0])
    top = max(parts[decode], key=parts[decode].get)
    caption = (
        f"In a {decode} step, the {top} takes {100 * parts[decode][top] / sum(parts[decode].values()):.0f}% "
        "of the time; tiny ops like RMSNorm cost far more than their arithmetic, because every launch "
        "has a fixed cost."
    )
    return fig, caption


def roofline_points(m1: Records, hw: Records, cfg: ModelConfig) -> tuple[plt.Figure, str]:
    bandwidth = measured_bandwidth(hw)
    peak = measured_peak_flops(hw).get("bf16")
    fig, ax = plt.subplots()
    intensity = np.logspace(-1, 4, 300)
    ax.plot(
        intensity,
        np.minimum(peak, bandwidth * intensity) / 1e12,
        color=BASELINE_GRAY,
        label="L4 roofline (M0)",
    )
    points = {"decode": [], "prefill": []}
    for m in metrics_of(m1, "decode_speed"):
        flops, nbytes = decode_flops_bytes(cfg, m["batch"], m["prompt_len"] + m["steps"] / 2)
        points["decode"].append((flops / nbytes, flops / (m["step_ms_median"] / 1e3), f"B={m['batch']}"))
    for m in metrics_of(m1, "prefill_speed"):
        flops, nbytes = prefill_flops_bytes(cfg, m["length"])
        points["prefill"].append((flops / nbytes, flops / (m["ms_median"] / 1e3), f"{m['length']} tok"))
    for kind, style in (("decode", series_style("fp8")), ("prefill", series_style("int4"))):
        pts = sorted(points[kind])
        ax.plot([p[0] for p in pts], [p[1] / 1e12 for p in pts], ls="-", label=f"nanoserve {kind}", **style)
        for x, y, text in pts:
            ax.annotate(text, (x, y / 1e12), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="arithmetic intensity (FLOPs/byte)",
        ylabel="TFLOP/s",
        title="nanoserve on the roofline",
    )
    ax.legend(fontsize=8)
    b1 = min(points["decode"])
    gap = min(peak, bandwidth * b1[0]) / b1[1]
    caption = (
        f"At batch 1, nanoserve decode reaches 1/{gap:.0f} of what the roofline allows at its intensity: "
        "launch overhead, not memory or compute, limits it."
    )
    return fig, caption


def kv_growth(cfg: ModelConfig, hw: Records) -> tuple[plt.Figure, str]:
    gpu_bytes = hw[0]["env"].get("gpu_memory_bytes", 24e9)
    free_for_kv = 0.9 * gpu_bytes - 2 * cfg.num_params()  # 10% headroom for activations, as vLLM's default
    context = np.linspace(0, 40960, 300)
    fig, ax = plt.subplots()
    markers = ["o", "s", "D", "^"]
    for batch, marker in zip((1, 8, 32, 128), markers, strict=True):
        kv = batch * context * cfg.kv_bytes_per_token() / 1e9
        ax.plot(context, kv, marker=marker, markevery=50, label=f"batch {batch}")
    ax.axhline(free_for_kv / 1e9, ls="--", color=BASELINE_GRAY, label="L4 memory left after weights")
    ax.set(
        xlabel="context length (tokens)",
        ylabel="KV cache (GB, BF16)",
        ylim=(0, 3 * free_for_kv / 1e9),
        title="KV cache growth: Qwen3-0.6B on an L4",
    )
    ax.legend()
    limit_32 = free_for_kv / (32 * cfg.kv_bytes_per_token())
    caption = (
        f"Each token costs {cfg.kv_bytes_per_token() / 1024:.0f} KiB of KV cache: "
        "at batch 32 the L4's free memory "
        f"fills at about {limit_32:,.0f} tokens of context."
    )
    return fig, caption


def attention_patterns(m1: Records) -> tuple[plt.Figure, str]:
    data = metrics_of(m1, "attention_maps")[0]
    layers = list(data["maps"])
    fig, axes = plt.subplots(
        1, len(layers) + 1, figsize=(4 * len(layers) + 4, 3.8), gridspec_kw={"wspace": 0.35}
    )
    for i, (ax, layer) in enumerate(zip(axes, layers, strict=False)):
        probs = np.array(data["maps"][layer])
        ax.imshow(probs, cmap="viridis", vmin=0, vmax=1)
        ticks = range(0, len(probs), max(1, len(probs) // 4))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set(title=f"layer {layer}, head {data['head']}", xlabel="key position")
        if i == 0:
            ax.set_ylabel("query position")
        ax.grid(False)
    sink = np.array(data["sink_by_layer"])
    axes[-1].bar(range(len(sink)), sink, color=OKABE_ITO["blue"])
    axes[-1].set(title="attention on token 0", xlabel="layer", ylabel="mean attention weight", ylim=(0, 1))
    fig.suptitle("What attention looks at", fontweight="bold")
    strongest = int(np.argmax(sink))
    caption = (
        f"Most layers pour attention onto the first token (an attention sink): layer {strongest} puts "
        f"{sink[strongest]:.0%} of its weight there on average "
        f"({sink.mean():.0%} across all {len(sink)} layers)."
    )
    return fig, caption


def _owner_grid(snapshot: dict[str, Any], num_blocks: int) -> np.ndarray:
    grid = np.full(num_blocks, -1)
    for seq, blocks in snapshot["block_tables"].items():
        grid[blocks] = int(seq)
    return grid


def block_table(m1: Records) -> tuple[plt.Figure, str, Any]:
    run = batching_summary(m1)
    n, log = run["num_blocks"], run["log"]
    cols = 8
    busy = [i for i, snapshot in enumerate(log) if snapshot["running"]]  # skip empty-pool snapshots
    picks = [busy[int(k)] for k in np.linspace(0, len(busy) - 1, 4)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    cmap = plt.get_cmap("tab10").with_extremes(under="#EEEEEE")
    for ax, i in zip(axes, picks, strict=True):
        grid = _owner_grid(log[i], n).reshape(-1, cols)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=9.99, aspect="equal")
        for (r, c), owner in np.ndenumerate(grid):
            if owner >= 0:
                ax.text(c, r, str(owner), ha="center", va="center", fontsize=7, color="white")
        ax.set(title=f"step {log[i]['step']}: {len(log[i]['running'])} running", xticks=[], yticks=[])
        ax.grid(False)
    fig.suptitle(
        f"Paged KV cache: {n} blocks of {run['block_size']} tokens (number = request id)", fontweight="bold"
    )
    # A block is "reused" when it's handed to a request after a different request owned it.
    owner_history: dict[int, list[int]] = {}
    for snapshot in log:
        for block, owner in enumerate(_owner_grid(snapshot, n)):
            history = owner_history.setdefault(block, [])
            if owner >= 0 and (not history or history[-1] != owner):
                history.append(int(owner))
    reuses = sum(max(len(h) - 1, 0) for h in owner_history.values())
    caption = (
        f"{len(set().union(*[s['block_tables'].keys() for s in log]))} requests shared {n} KV blocks over "
        f"{run['steps']} steps; blocks freed by finished requests were handed to new ones {reuses} times."
    )
    return fig, caption, _block_table_animation(run)


def _block_table_animation(run: dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    n, cols, log = run["num_blocks"], 8, run["log"]
    grids = [_owner_grid(s, n).reshape(-1, cols) for s in log]
    frames = [
        go.Frame(
            data=[
                go.Heatmap(
                    z=np.where(g < 0, np.nan, g),
                    text=np.where(g < 0, "", g.astype(str)),
                    texttemplate="%{text}",
                    zmin=0,
                    zmax=9,
                    colorscale="Turbo",
                    showscale=False,
                )
            ],
            name=str(s["step"]),
            layout=go.Layout(title=f"step {s['step']}: running {s['running']}, waiting {s['waiting']}"),
        )
        for g, s in zip(grids, log, strict=True)
    ]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Paged KV cache: block ownership per step",
        yaxis={"autorange": "reversed", "visible": False},
        xaxis={"visible": False},
        updatemenus=[
            {"type": "buttons", "buttons": [{"label": "Play", "method": "animate", "args": [None]}]}
        ],
        sliders=[{"steps": [{"label": f.name, "method": "animate", "args": [[f.name]]} for f in frames]}],
    )
    return fig


def make_all(m1: Records, hw: Records, cfg: ModelConfig, out_dir: str | Path) -> list[Path]:
    """Render every M1 figure. `hw` is the M0 probe run (for the roofline and GPU memory)."""
    apply_style()
    written: list[Path] = []
    for name, (fig, caption) in {
        "m1_decode_scaling": decode_scaling(m1),
        "m1_anatomy": anatomy(m1),
        "m1_roofline": roofline_points(m1, hw, cfg),
        "m1_kv_growth": kv_growth(cfg, hw),
        "m1_attention": attention_patterns(m1),
    }.items():
        written += save_figure(fig, name, caption, out_dir)
    fig, caption, animation = block_table(m1)
    written += save_figure(fig, "m1_block_table", caption, out_dir)
    written.append(save_html(animation, "m1_block_table", out_dir))
    return written
