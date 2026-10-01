"""M5 figures: KV-cache policies, FP8 KV and prefix caching. Each returns (figure, data-computed caption)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, ScalarFormatter

from fastserve.kv.prefix_cache import replay
from fastserve.kv.sizing import KVSpec, bytes_per_token
from fastserve.report.m5 import (
    LARGE,
    POLICY_LABELS,
    SMALL,
    _median,
    _one,
    held,
    needle,
    policy_spec,
    serving,
    start,
    summary_stat,
)
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, apply_style, save_figure

Records = list[dict[str, Any]]
POLICY_STYLE = {  # BF16 always gray; the KIVI policies share the KV-quantization green
    "bf16": (BASELINE_GRAY, "o", "-"),
    "fp8": (OKABE_ITO["blue"], "s", "-"),
    "int8": (OKABE_ITO["sky_blue"], "D", "-"),
    "int8-kivi": (OKABE_ITO["sky_blue"], "d", "--"),
    "int4-token": (OKABE_ITO["orange"], "^", "-"),
    "int4-token-rot": (OKABE_ITO["yellow"], "v", "-"),
    "int4-kivi": (OKABE_ITO["green"], "P", "-"),
    "int2-kivi": (OKABE_ITO["green"], "X", "--"),
    "streaming": (OKABE_ITO["black"], "*", ":"),
}


def needle_heatmaps(m5: Records, policies: list[dict[str, Any]]) -> tuple[plt.Figure, str]:
    """M2's needle grid per KV policy in nanoserve, plus vLLM with an FP8 cache: green = found."""
    panels = [(POLICY_LABELS.get(p["name"], p["name"]), needle(m5, SMALL, p["name"])) for p in policies]
    panels.append(("vLLM, FP8 KV", _one(m5, "m5_vllm_needle", model=SMALL)))
    panels = [(title, m) for title, m in panels if m]
    cols = 3
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.3 * rows), squeeze=False)
    lengths = sorted({c["length"] for _, m in panels for c in m["cells"]})
    depths = sorted({c["depth"] for _, m in panels for c in m["cells"]})
    for ax, (title, m) in zip(axes.flat, panels, strict=False):
        grid = np.zeros((len(depths), len(lengths)))
        count = np.zeros_like(grid)
        for c in m["cells"]:
            i, j = depths.index(c["depth"]), lengths.index(c["length"])
            grid[i, j] += c["passed"]
            count[i, j] += 1
        share = grid / np.maximum(count, 1)
        ax.imshow(share, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(depths)):
            for j in range(len(lengths)):
                ax.text(j, i, f"{int(grid[i, j])}/{int(count[i, j])}", ha="center", va="center", fontsize=7)
        ax.set_xticks(
            range(len(lengths)), [f"{n // 1000}k" if n >= 1000 else str(n) for n in lengths], fontsize=8
        )
        ax.set_yticks(range(len(depths)), [f"{d:.0%}" for d in depths], fontsize=8)
        ax.set_title(f"{title}: {m['pass_rate']:.0%}", fontsize=9)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("needle depth")
    for ax in axes[-1, :]:
        ax.set_xlabel("context (tokens)")
    fig.tight_layout()
    lost = [f"{title} ({m['pass_rate']:.0%})" for title, m in panels if m["pass_rate"] < 0.95]
    caption = (
        "Every policy keeps the needle except " + ", ".join(lost) + "."
        if lost
        else "Every KV policy keeps the needle at every length and depth."
    )
    return fig, caption


def key_value_channels(m5: Records, layer: str = "14") -> tuple[plt.Figure, str]:
    """Per-channel max |K| and |V| in one layer (every head a line), and the outlier ratio in every layer."""
    stats = _one(m5, "m5_kv_stats", model=SMALL)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, kind in zip(axes[:2], ("keys", "values"), strict=True):
        for head in stats["profiles"][layer][kind]:
            ax.plot(
                head, lw=0.8, alpha=0.7, color=OKABE_ITO["blue"] if kind == "keys" else OKABE_ITO["orange"]
            )
        ax.set(
            title=f"{kind.capitalize()}, layer {layer} (one line per KV head)",
            xlabel="channel",
            ylabel="max |value| over tokens",
        )
    ax = axes[2]
    layers = range(len(stats["key_ratio"]))
    ax.plot(layers, stats["key_ratio"], marker="o", color=OKABE_ITO["blue"], label="keys")
    ax.plot(layers, stats["value_ratio"], marker="s", ls="--", color=OKABE_ITO["orange"], label="values")
    ax.set(
        yscale="log", xlabel="layer", ylabel="largest ÷ median channel", title="Outlier channels, every layer"
    )
    ax.yaxis.set_major_formatter(ScalarFormatter())  # plain numbers, not 6×10¹
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8)
    fig.tight_layout()
    k, v = _median(stats["key_ratio"]), _median(stats["value_ratio"])
    caption = (
        f"Keys have outlier channels, values don't: in the median layer the largest key channel is {k:.1f}× "
        f"the median one, against {v:.1f}× for values, so one scale per token wastes the grid on keys."
    )
    return fig, caption


def concurrency_vs_context(
    m5: Records, policies: list[dict[str, Any]], cfg: Any, kv_bytes: float
) -> tuple[plt.Figure, str]:
    """How many sequences of each length fit in the L4's KV budget, per format; vLLM's points on top."""
    fig, ax = plt.subplots(figsize=(8, 5))
    contexts = np.geomspace(512, 40960, 60)
    for p in policies:
        if p["name"] in ("int4-token-rot",):  # same size as int4-token
            continue
        spec = policy_spec(p["name"], policies)
        color, marker, ls = POLICY_STYLE.get(p["name"], (BASELINE_GRAY, "o", "-"))
        fits = [kv_bytes / (spec.kept_tokens(int(c)) * bytes_per_token(cfg, spec)) for c in contexts]
        ax.plot(contexts, fits, color=color, ls=ls, label=POLICY_LABELS.get(p["name"], p["name"]))
    measured = []
    for label, color in (("bf16kv", BASELINE_GRAY), ("fp8kv", OKABE_ITO["blue"])):
        s, cap = start(m5, SMALL, label), held(m5, SMALL, label, "capacity")
        if s and s.get("max_concurrency"):
            ax.scatter(
                [40960], [s["max_concurrency"]], color=color, marker="o", s=60, zorder=3, edgecolor="black"
            )
        if cap:
            ax.scatter(
                [4096 + 256], [cap["running"]], color=color, marker="D", s=60, zorder=3, edgecolor="black"
            )
            measured.append((label, cap["running"]))
    ax.scatter([], [], color="white", edgecolor="black", marker="D", label="vLLM: running, capacity workload")
    ax.scatter([], [], color="white", edgecolor="black", marker="o", label="vLLM: reported max concurrency")
    ax.set(xscale="log", yscale="log", xlabel="tokens per sequence", ylabel="sequences that fit at once")
    ax.legend(fontsize=7, ncols=2)
    bf16 = KVSpec("BF16")
    kivi = policy_spec("int4-kivi", policies)
    gain = bytes_per_token(cfg, bf16) / bytes_per_token(cfg, kivi)
    caption = (
        f"In the L4's {kv_bytes / 2**30:.1f} GiB KV budget, FP8 fits twice the sequences of BF16 and INT4 "
        f"KIVI {gain:.1f}×; only eviction keeps the count flat as contexts grow"
    )
    if len(measured) == 2:
        caption += (
            f". With 4k-token prompts vLLM ran {measured[0][1]:.0f} (BF16) vs {measured[1][1]:.0f} (FP8)"
        )
    return fig, caption + "."


def radix_tree(specs: list[Any], conversations: int) -> tuple[plt.Figure, str]:
    """The multi-turn workload's prompts as a radix tree: one row per conversation, one block per tree node
    spanning the conversations that share it; color = how many requests reused it."""
    prompts = [s.prompt for s in specs]
    cache, cached = replay(prompts)
    covers: dict[int, set[int]] = {}
    starts: dict[int, int] = {}
    for i, prompt in enumerate(prompts):  # walk each prompt down the final tree: which nodes it uses
        node, depth = cache.root, 0
        while depth < len(prompt):
            child = node.children[prompt[depth]]
            covers.setdefault(child.id, set()).add(i % conversations)
            starts[child.id] = depth
            node, depth = child, depth + len(child.tokens)
    apps = {c: tuple(prompts[c][:16]) for c in range(conversations)}  # conversations of one app share a start
    order = sorted(range(conversations), key=lambda c: (apps[c], c))
    row = {c: r for r, c in enumerate(order)}
    fig, ax = plt.subplots(figsize=(11, 6))
    norm = LogNorm(vmin=1, vmax=max(n.hits + 1 for n in cache.nodes()))
    cmap = plt.get_cmap("viridis")
    for node in cache.nodes():  # one outlined block per node, across every conversation that shares it
        rows = sorted(row[c] for c in covers.get(node.id, ()))
        if not rows:
            continue
        block = plt.Rectangle(
            (starts[node.id], rows[0]),
            len(node.tokens),
            rows[-1] - rows[0] + 1,
            facecolor=cmap(norm(node.hits + 1)),
            edgecolor="white",
            lw=0.6,
        )
        ax.add_patch(block)
    groups: dict[tuple, list[int]] = {}
    for c in order:
        groups.setdefault(apps[c], []).append(row[c])
    ticks = [sum(rows) / len(rows) + 0.5 for rows in groups.values()]
    ax.set_yticks(ticks, [f"app {i + 1}" for i in range(len(groups))])
    ax.set(
        xlim=(0, max(len(p) for p in prompts)),
        ylim=(conversations, 0),
        xlabel="token position in the prompt",
        ylabel="conversations, grouped by app",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="requests that reused the node + 1"
    )
    total = sum(len(p) for p in prompts)
    caption = (
        f"{len(prompts)} prompts ({total:,} tokens) collapse into {cache.size:,} distinct tokens: each app's "
        f"system prompt is computed once, and {sum(cached) / total:.0%} of all prompt tokens could come from "
        "the cache."
    )
    return fig, caption


def prefix_ttft(m5: Records, workload: str = "multi_turn") -> tuple[plt.Figure, str]:
    """Per request: TTFT against the share of its prompt found in the cache (caching on), and off for
    contrast; below, the prefill tokens actually computed over the session."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    lines = []
    for col, model in enumerate((SMALL, LARGE)):
        totals = {}
        for label, color, marker in (
            ("prefix-off", BASELINE_GRAY, "o"),
            ("prefix-on", OKABE_ITO["purple"], "P"),
        ):
            m = serving(m5, model, label, workload)
            if not m:
                continue
            cols = {c: i for i, c in enumerate(m["requests"]["columns"])}
            rows = sorted(m["requests"]["rows"], key=lambda r: r[cols["id"]])
            share = [(r[cols["cached_tokens"]] or 0) / r[cols["prompt_len"]] for r in rows]
            ttft = [1e3 * (r[cols["first_token"]] - r[cols["sent"]]) for r in rows]
            axes[0, col].scatter(
                share,
                ttft,
                s=12,
                color=color,
                marker=marker,
                alpha=0.7,
                label=f"caching {label.split('-')[1]}",
            )
            computed = np.cumsum([r[cols["prompt_len"]] - (r[cols["cached_tokens"]] or 0) for r in rows])
            axes[1, col].plot(computed, color=color, label=f"caching {label.split('-')[1]}")
            totals[label] = computed[-1] if len(computed) else None
        axes[0, col].set(
            title=model.split("/")[-1],
            xlabel="share of the prompt found in the cache",
            ylabel="time to first token (ms)",
            yscale="log",
        )
        axes[1, col].set(xlabel="request", ylabel="prompt tokens prefilled (cumulative)")
        axes[1, col].legend(fontsize=8)
        axes[0, col].legend(fontsize=8)
        on = summary_stat(m5, model, "prefix-on", workload, "ttft_ms")
        off = summary_stat(m5, model, "prefix-off", workload, "ttft_ms")
        if on and off and totals.get("prefix-on") and totals.get("prefix-off"):
            lines.append(
                f"{model.split('/')[-1]}: TTFT p50 {off:.0f} → {on:.0f} ms, prefill "
                f"{1 - totals['prefix-on'] / totals['prefix-off']:.0%} smaller"
            )
    fig.tight_layout()
    return fig, "Prefix caching on the multi-turn workload: " + "; ".join(lines) + "."


SETUPS = {  # what each server is, in words
    "bf16kv": "BF16",
    "bf16kv-flashinfer": "BF16 on FlashInfer",
    "fp8kv": "FP8 KV",
    "fp8w-fp8kv": "FP8 weights + FP8 KV",
    "prefix-off": "no caching",
    "prefix-on": "prefix caching",
}


def waterfall_v2(m5: Records, dollars_per_hour: float = 0.80) -> tuple[plt.Figure, str]:
    """$ per 1M output tokens: BF16, then each M5 change as a step, in the regime where it applies.

    FP8 KV on the L4 also switches the attention kernel to FlashInfer, so that switch gets its own step:
    the FP8-KV step is then what storing the cache in 8 bits buys on the same kernel.
    """
    fp8 = [
        ("bf16kv", "BF16"),
        ("bf16kv-flashinfer", "FlashInfer kernel"),
        ("fp8kv", "+ FP8 KV"),
        ("fp8w-fp8kv", "+ FP8 weights"),
    ]
    regimes = [
        ("saturated server", "saturation", fp8),
        ("96 users, 4k prompts", "capacity", fp8),
        ("multi-turn chat", "multi_turn", [("prefix-off", "no caching"), ("prefix-on", "+ prefix caching")]),
    ]
    fig, axes = plt.subplots(len(regimes), 2, figsize=(11, 10))
    cheapest: dict[tuple[str, str], tuple[str, float]] = {}
    kernel: dict[tuple[str, str], float] = {}
    for row, (title, workload, steps) in enumerate(regimes):
        for col, model in enumerate((SMALL, LARGE)):
            ax, costs = axes[row, col], []
            for label, name in steps:
                p = held(m5, model, label, workload) if workload != "multi_turn" else None
                tok_s = (
                    p["output_tok_s"] if p else summary_stat(m5, model, label, workload, "output_throughput")
                )
                if tok_s:
                    costs.append((label, name, dollars_per_hour / (tok_s * 3600) * 1e6))
            if not costs:
                ax.axis("off")
                continue
            base = costs[0][2]
            for i, (_label, _name, cost) in enumerate(costs):
                if i == 0:
                    ax.bar(i, base, color=BASELINE_GRAY)
                    continue
                prev = costs[i - 1][2]
                color = OKABE_ITO["green"] if cost < prev else OKABE_ITO["vermillion"]
                ax.bar(i, cost - prev, bottom=prev, color=color)
                ax.annotate(
                    f"{cost / prev - 1:+.0%}", (i, max(cost, prev)), ha="center", va="bottom", fontsize=8
                )
            ax.set_xticks(range(len(costs)), [name for _, name, _ in costs], fontsize=8)
            ax.set(title=f"{model.split('/')[-1]}, {title}", ylabel="$ per 1M output tokens")
            ax.set_ylim(0, max(c for _, _, c in costs) * 1.15)
            label, _, cost = min(costs, key=lambda c: c[2])
            cheapest[model, title] = (SETUPS[label], cost / base - 1)
            by_label = {label: cost for label, _, cost in costs}
            if "bf16kv-flashinfer" in by_label:
                kernel[model, title] = by_label["bf16kv-flashinfer"] / base - 1
    fig.tight_layout()
    parts = []
    for title, _, _ in regimes:
        if (SMALL, title) not in cheapest:
            continue
        setup, change = cheapest[SMALL, title]
        note = setup
        if (SMALL, title) in kernel:
            note += f"; the FlashInfer kernel alone {kernel[SMALL, title]:+.0%}"
        parts.append(f"{title} {change:+.0%} ({note})")
    caption = "Waterfall v2, Qwen3-0.6B, the cheapest setup against BF16: " + ", ".join(parts) + "."
    return fig, caption


def make_all(
    m5: Records, policies: list[dict[str, Any]], cfg: Any, workloads: dict[str, Any], out_dir: str | Path
) -> list[Path]:
    from fastserve.serving.workloads import Workload

    apply_style()
    bf16_start = start(m5, SMALL, "bf16kv")
    kv_bytes = (bf16_start or {}).get("kv_cache_memory_gib", 18.44) * 2**30
    multi = Workload.from_config("multi_turn", workloads["multi_turn"])
    figures = {
        "m5_needle": lambda: needle_heatmaps(m5, policies),
        "m5_key_value_channels": lambda: key_value_channels(m5),
        "m5_concurrency": lambda: concurrency_vs_context(m5, policies, cfg, kv_bytes),
        "m5_radix_tree": lambda: radix_tree(multi.requests(), multi.num_requests // multi.turns),
        "m5_prefix_ttft": lambda: prefix_ttft(m5),
        "m5_waterfall": lambda: waterfall_v2(m5),
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
