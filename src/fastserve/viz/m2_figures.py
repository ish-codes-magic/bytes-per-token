"""M2 figures: stock vLLM under load. Each function returns (figure, caption computed from data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fastserve.report.m2 import LARGE, SMALL, knee_rate, peak_throughput, runs
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, apply_style, save_figure

Records = list[dict[str, Any]]
MODEL_STYLE = {
    SMALL: {"color": OKABE_ITO["blue"], "marker": "o", "label": "Qwen3-0.6B"},
    LARGE: {"color": OKABE_ITO["vermillion"], "marker": "s", "label": "Qwen3-1.7B"},
}


def _per_request(row_table: dict[str, Any]) -> dict[str, np.ndarray]:
    cols = {name: i for i, name in enumerate(row_table["columns"])}
    rows = np.array(row_table["rows"], dtype=float)
    out = {name: rows[:, i] for name, i in cols.items()}
    out["ttft_ms"] = (out["first_token"] - out["sent"]) * 1e3
    decode_tokens = np.maximum(out["output_tokens"] - 1, 1)
    out["tpot_ms"] = (out["finished"] - out["first_token"]) / decode_tokens * 1e3
    return out


def pareto(records: Records) -> tuple[plt.Figure, str]:
    fig, ax = plt.subplots()
    for model, style in MODEL_STYLE.items():
        pts = sorted(
            (m["summary"]["output_throughput"], m["summary"]["tpot_ms"]["p50"], m["load"]["rate"])
            for m in runs(records, model=model, workload="throughput", mode="open")
            if m["summary"].get("tpot_ms")
        )
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **style)
        for x, y, rate in pts:
            ax.annotate(f"{rate:g}/s", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set(
        xlabel="output throughput (tokens/s)",
        ylabel="TPOT p50 (ms)",
        ylim=(0, None),
        title="Latency vs throughput as load rises (labels: arrival rate)",
    )
    ax.legend()
    peak = peak_throughput(records, SMALL)
    tpots = [
        m["summary"]["tpot_ms"]["p50"] for m in runs(records, model=SMALL, workload="throughput", mode="open")
    ]
    caption = (
        f"Qwen3-0.6B reaches {peak:,.0f} output tokens/s; as load rises its TPOT p50 grows from "
        f"{min(tpots):.1f} to {max(tpots):.1f} ms, the price of bigger batches."
    )
    return fig, caption


def goodput(records: Records) -> tuple[plt.Figure, str]:
    fig, ax = plt.subplots()
    top = 0.0
    for model, style in MODEL_STYLE.items():
        rows = sorted(
            runs(records, model=model, workload="throughput", mode="open"), key=lambda m: m["load"]["rate"]
        )
        if not rows:
            continue
        rate = np.array([m["load"]["rate"] for m in rows])
        good = np.array([m["summary"].get("goodput_requests", 0.0) for m in rows])
        done = np.array([m["summary"].get("request_throughput", 0.0) for m in rows])
        ax.plot(rate, good, **style)
        ax.plot(rate, done, ls=":", color=style["color"], label=f"{style['label']}: all completed")
        knee = knee_rate(records, model)
        if knee:
            ax.axvline(knee, color=style["color"], lw=0.8, alpha=0.5)
        top = max(top, rate.max())
    ax.plot([0, top], [0, top], ls="--", color=BASELINE_GRAY, lw=1, label="offered load")
    ax.set(
        xlabel="offered load (requests/s, Poisson)",
        ylabel="requests/s",
        title="Goodput: requests within the SLO",
    )
    ax.legend(fontsize=8)
    knee = knee_rate(records, SMALL)
    caption = (
        f"Qwen3-0.6B keeps at least 90% of requests within the SLO up to {knee:g} req/s; past that "
        "the queue grows and goodput falls even though requests keep completing."
    )
    return fig, caption


def cdfs(records: Records) -> tuple[plt.Figure, str]:
    rows = sorted(
        runs(records, model=SMALL, workload="throughput", mode="open"), key=lambda m: m["load"]["rate"]
    )
    picks = [rows[0], rows[len(rows) // 2], rows[-1]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    colors = [OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["vermillion"]]
    styles = ["-", "--", ":"]
    for m, color, ls in zip(picks, colors, styles, strict=True):
        data = _per_request(m["requests"])
        for ax, key in ((left, "ttft_ms"), (right, "tpot_ms")):
            values = np.sort(data[key])
            ax.plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                color=color,
                ls=ls,
                label=f"{m['load']['rate']:g} req/s",
            )
    left.set(
        xscale="log", xlabel="TTFT (ms, log)", ylabel="fraction of requests", title="Time to first token"
    )
    right.set(xlabel="TPOT (ms)", title="Time per output token")
    left.legend()
    fig.suptitle("Qwen3-0.6B latency distributions (throughput workload)", fontweight="bold")
    low, high = picks[0]["summary"]["ttft_ms"], picks[-1]["summary"]["ttft_ms"]
    caption = (
        f"TTFT p99 goes from {low['p99']:,.0f} ms at {picks[0]['load']['rate']:g} req/s to "
        f"{high['p99']:,.0f} ms at {picks[-1]['load']['rate']:g} req/s: averages hide the queueing tail."
    )
    return fig, caption


def swimlane(records: Records, n: int = 60) -> tuple[plt.Figure, str]:
    """Requests from the middle of the most loaded run, where the queue has built up."""
    from matplotlib.patches import Patch

    run = max(runs(records, model=SMALL, workload="throughput", mode="open"), key=lambda m: m["load"]["rate"])
    data = _per_request(run["requests"])
    by_arrival = np.argsort(data["sent"])
    middle = len(by_arrival) // 2
    order = by_arrival[max(0, middle - n // 2) : middle + n // 2]
    fig, ax = plt.subplots(figsize=(9, 5))
    for lane, i in enumerate(order):
        sent, first, end = data["sent"][i], data["first_token"][i], data["finished"][i]
        ax.barh(lane, first - sent, left=sent, color=OKABE_ITO["orange"], height=0.8)
        ax.barh(lane, end - first, left=first, color=OKABE_ITO["blue"], height=0.8)
    handles = [
        Patch(color=OKABE_ITO["orange"], label="waiting + prefill (until the first token)"),
        Patch(color=OKABE_ITO["blue"], label="decoding"),
    ]
    # Below the x-axis label: inside the axes every corner is covered by some bar.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, frameon=False)
    ax.set(
        xlabel="time since the run started (s)",
        ylabel="request (by arrival)",
        title=f"{len(order)} requests from the middle of a {run['load']['rate']:g} req/s run",
    )
    ax.invert_yaxis()
    ax.grid(False)
    # How many requests decode at the same moment: sweep over start/end events.
    events = sorted([(data["first_token"][i], 1) for i in order] + [(data["finished"][i], -1) for i in order])
    running = peak_running = 0
    for _, delta in events:
        running += delta
        peak_running = max(peak_running, running)
    waits = data["ttft_ms"][order]
    rate, longest, median = run["load"]["rate"], waits.max(), np.median(waits)
    caption = (
        f"Mid-run at {rate:g} req/s, up to {peak_running} of these {len(order)} requests decode together, "
        f"and waits for a first token reach {longest:,.0f} ms (median {median:,.0f} ms)."
    )
    return fig, caption


def long_context(records: Records) -> tuple[plt.Figure, str]:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    lengths = {"long_8k": 8192, "long_16k": 16384, "long_32k": 32768}
    for model, style in MODEL_STYLE.items():
        pts = []
        for workload, n in lengths.items():
            for m in runs(records, model=model, workload=workload):
                pts.append((n, m["summary"]["ttft_ms"]["p50"], m["summary"]["tpot_ms"]["p50"]))
        chat_rows = runs(records, model=model, workload="chat")
        if chat_rows:
            s = chat_rows[0]["summary"]
            pts.append((100, s["ttft_ms"]["p50"], s["tpot_ms"]["p50"]))
        pts.sort()
        if not pts:
            continue
        left.plot([p[0] for p in pts], [p[1] for p in pts], **style)
        right.plot([p[0] for p in pts], [p[2] for p in pts], **style)
    left.set(xscale="log", yscale="log", xlabel="prompt tokens", ylabel="TTFT p50 (ms)", title="Prefill")
    right.set(xscale="log", xlabel="context tokens", ylabel="TPOT p50 (ms)", ylim=(0, None), title="Decode")
    left.legend()
    fig.suptitle("Long context: one user", fontweight="bold")
    chat_s = runs(records, model=SMALL, workload="chat")[0]["summary"]
    long_s = runs(records, model=SMALL, workload="long_32k")[0]["summary"]
    ttft, slowdown = long_s["ttft_ms"]["p50"], long_s["tpot_ms"]["p50"] / chat_s["tpot_ms"]["p50"]
    caption = (
        f"With a 32k-token prompt, Qwen3-0.6B takes {ttft:,.0f} ms to its first token, and each later token "
        f"is {slowdown:.1f}× slower than in chat: every step re-reads the whole KV cache."
    )
    return fig, caption


FIGURES = {  # figures drawn from the baseline sweep alone
    "m2_pareto": pareto,
    "m2_goodput": goodput,
    "m2_cdfs": cdfs,
    "m2_swimlane": swimlane,
    "m2_long_context": long_context,
}


def make_all(records: Records, out_dir: str | Path) -> list[Path]:
    apply_style()
    written: list[Path] = []
    for name, build in FIGURES.items():
        fig, caption = build(records)
        written += save_figure(fig, name, caption, out_dir)
    return written


def needle_heatmaps(quality_records: Records) -> tuple[plt.Figure, str]:
    from fastserve.report.m2 import quality_by_model

    q = quality_by_model(quality_records)
    models = [m for m in (SMALL, LARGE) if "needle_cells" in q.get(m, {})]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models) + 1, 4), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        cells = q[model]["needle_cells"]
        lengths = sorted({c["length"] for c in cells})
        depths = sorted({c["depth"] for c in cells})
        grid = np.zeros((len(depths), len(lengths)))
        counts = np.zeros_like(grid)
        for c in cells:
            i, j = depths.index(c["depth"]), lengths.index(c["length"])
            grid[i, j] += c["passed"]
            counts[i, j] += 1
        rate = grid / np.maximum(counts, 1)
        ax.imshow(rate, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for (i, j), passed_count in np.ndenumerate(grid):
            ax.text(
                j,
                i,
                f"{int(passed_count)}/{int(counts[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if rate[i, j] > 0.6 else "white",
            )
        ax.set_xticks(range(len(lengths)), [f"{n // 1024}k" if n >= 1024 else str(n) for n in lengths])
        ax.set_yticks(range(len(depths)), [f"{d:.0%}" for d in depths])
        ax.set(xlabel="context length (tokens)", ylabel="needle depth", title=model.split("/")[-1])
        ax.grid(False)
    fig.suptitle("Needle in a haystack (BF16): passed / tried", fontweight="bold")
    rates = {model.split("/")[-1]: q[model]["needle"] for model in models}
    if len(set(rates.values())) == 1:
        who = "both models retrieve" if len(rates) > 1 else f"{next(iter(rates))} retrieves"
        caption = f"In BF16, {who} the needle in {next(iter(rates.values())):.0f}% of cells"
    else:
        name, rate = min(rates.items(), key=lambda kv: kv[1])
        caption = f"In BF16 the weaker model ({name}) retrieves the needle in {rate:.0f}% of cells"
    caption += " up to 32k tokens: the bar that KV-cache compression must not lower."
    return fig, caption


def make_quality(quality_records: Records, out_dir: str | Path) -> list[Path]:
    apply_style()
    fig, caption = needle_heatmaps(quality_records)
    return save_figure(fig, "m2_needle", caption, out_dir)


def saturation(records: Records, configs: dict[str, Any], bandwidth: float) -> tuple[plt.Figure, str]:
    """The engine held full, seen through its own counters, vs the memory-bound ceiling at each moment."""
    from fastserve.report.m2 import (
        drained,
        kv_capacity,
        memory_bound_step_s,
        memory_efficiency,
        plateau,
        saturation_run,
    )

    m, p = saturation_run(records, SMALL, "saturation")
    cfg = configs[SMALL]
    capacity = kv_capacity(records, SMALL)
    table = m["server_timeline"]
    keys = ("t", "running", "waiting", "kv_usage", "generation_tokens")
    idx = [table["columns"].index(k) for k in keys]
    rows = np.array([[row[i] for i in idx] for row in table["rows"] if all(row[i] is not None for i in idx)])
    t, running, waiting, kv_usage, generated = rows.T
    k = max(1, round(1.0 / np.median(np.diff(t))))  # rates over ~1 s: counters are too coarse per sample
    rate_t, rate = (t[k:] + t[:-k]) / 2, (generated[k:] - generated[:-k]) / (t[k:] - t[:-k])
    ceiling = running / np.array([memory_bound_step_s(u * capacity, cfg, bandwidth) for u in kv_usage])

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), height_ratios=[3, 2])
    queued = t[waiting > 0]
    for ax in (top, bottom):
        ax.axvspan(queued.min(), queued.max(), color=OKABE_ITO["yellow"], alpha=0.15, lw=0)
    top.plot(t, ceiling, color=BASELINE_GRAY, ls="--", label="memory-bound ceiling (live KV cache + weights)")
    top.plot(rate_t, rate, color=OKABE_ITO["blue"], label="measured (server counters, 1 s windows)")
    top.set(ylabel="output tokens/s", title="Qwen3-0.6B held at saturation (shaded: requests waiting)")
    top.set_ylim(bottom=0)
    top.legend(loc="lower center", fontsize=8)
    bottom.plot(t, running, color=OKABE_ITO["blue"], label="running")
    bottom.plot(t, waiting, color=OKABE_ITO["orange"], label="waiting")
    bottom.set(xlabel="time since the load started (s)", ylabel="requests")
    kv_axis = bottom.twinx()
    kv_axis.plot(t, 100 * kv_usage, color=OKABE_ITO["green"], ls=":", label="KV cache used (%)")
    kv_axis.set(ylabel="KV cache used (%)", ylim=(0, 105))
    kv_axis.grid(False)
    handles = bottom.get_legend_handles_labels()[0] + kv_axis.get_legend_handles_labels()[0]
    bottom.legend(handles=handles, loc="center right", fontsize=8)

    caption = (
        f"Held full for {p['seconds']:.0f} s, Qwen3-0.6B decodes {p['output_tok_s']:,.0f} tokens/s with "
        f"{p['running']:.0f} sequences running, at {memory_efficiency(p, capacity, cfg, bandwidth):.0%} of "
        "the memory-bound ceiling"
    )
    if drain := plateau(m, drained):
        caption += (
            f"; once the queue drains and no new prompts arrive, steps run at "
            f"{memory_efficiency(drain, capacity, cfg, bandwidth):.0%} of it"
        )
    caption += "."
    return fig, caption


def make_saturation(
    records: Records, configs: dict[str, Any], bandwidth: float, out_dir: str | Path
) -> list[Path]:
    apply_style()
    fig, caption = saturation(records, configs, bandwidth)
    return save_figure(fig, "m2_saturation", caption, out_dir)
