"""M8 figures: the full stack. Each returns (figure, caption computed from the data)."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from fastserve.perfmodel.serving import Calibration, Hardware
from fastserve.report import m8 as report
from fastserve.report.m8 import (
    BASE,
    FULL,
    LADDER,
    LARGE,
    SMALL,
    TECHNIQUES,
    WORKLOAD_LABELS,
    WORKLOADS,
    best,
    dollars,
    interaction,
    pairs,
    prediction_errors,
    repeat_differences,
    speedup,
    step_ms,
    tok_s,
    without,
)
from fastserve.serving.ablation import letters
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, SERIES, apply_style, save_figure

Records = list[dict[str, Any]]
STEP_COLOR = {  # one color per technique, as everywhere in the repo
    "w": SERIES["fp8"][0],
    "k": SERIES["kv_quant"][0],
    "p": SERIES["prefix_cache"][0],
    "s": SERIES["speculative"][0],
}
SHORT = {
    "m8_latency": "latency\n1 user",
    "spec_mixed": "busy\n64 users",
    "capacity": "capacity\n96 users × 4k",
    "multi_turn": "multi-turn\nshared prefixes",
    "long_32k": "long\n32k tokens",
}
MARKERS = {"m8_latency": "o", "spec_mixed": "s", "capacity": "^", "multi_turn": "D", "long_32k": "P"}


def _name(model: str) -> str:
    return model.split("/")[-1]


def _short(workload: str) -> str:
    """ "Latency (1 user, real prompts)" → "latency"."""
    return WORKLOAD_LABELS[workload].split(" (")[0].lower()


# ---- 1. the final waterfall --------------------------------------------------------------------------------


def final_waterfall(
    m8: Records, dollars_per_hour: float, projected: dict[str, float] | None = None
) -> tuple[plt.Figure, str]:
    """$ per 1M tokens down the ladder, per workload and model. Each floating bar is one technique's step.

    A step that made serving dearer is hatched. After the full stack comes the best *measured* stack, named,
    when it is a different one. `projected` maps a model to the tokens/s M7's kernel projection gives for
    the long workload: drawn as an outline, because it was never run in vLLM.
    """
    models = [m for m in (LARGE, SMALL) if tok_s(m8, m, BASE, "m8_latency")]
    workloads = [w for w in WORKLOADS if tok_s(m8, models[0], BASE, w)]
    fig, axes = plt.subplots(len(models), len(workloads), figsize=(3.5 * len(workloads), 4.1 * len(models)))
    axes = np.atleast_2d(axes)
    gains, full_gains = {}, {}
    for row, model in zip(axes, models, strict=True):
        for ax, workload in zip(row, workloads, strict=True):
            costs = [dollars(tok_s(m8, model, label, workload), dollars_per_hour) for label in LADDER]
            if None in costs:
                ax.axis("off")
                continue
            ax.bar(0, costs[0], color=BASELINE_GRAY)
            for i, letter in enumerate(TECHNIQUES, start=1):
                lo, hi = sorted((costs[i - 1], costs[i]))
                rising = costs[i] > costs[i - 1]
                ax.bar(i, hi - lo, bottom=lo, color=STEP_COLOR[letter], hatch="xx" if rising else None)
                change = costs[i] / costs[i - 1] - 1
                if abs(change) >= 0.005:
                    ax.annotate(f"{change:+.0%}", (i, hi), ha="center", va="bottom", fontsize=7.5)
            ax.bar(len(LADDER), costs[-1], color=OKABE_ITO["black"])
            ticks = [
                "stock\nBF16",
                "FP8\nweights",
                "FP8\nKV",
                "prefix\ncache",
                "spec.\ndecode",
                "full\nstack",
            ]
            top = best(m8, model, workload, allow_int4=False)
            position = len(LADDER) + 1
            if top and top[0] != FULL:
                ax.bar(position, dollars(top[1], dollars_per_hour), color=OKABE_ITO["sky_blue"])
                ticks.append(f"best:\n{top[0]}")
                position += 1
            extra = (projected or {}).get(model) if workload == "long_32k" else None
            if extra:
                cost = dollars(extra, dollars_per_hour)
                ax.bar(position, cost, color="white", edgecolor=OKABE_ITO["black"], hatch="///")
                ticks.append("kernel 2\n(projected)")
            ax.set_xticks(range(len(ticks)), ticks, fontsize=6.5)
            ax.set_title(f"{_name(model)}, {SHORT[workload].replace(chr(10), ', ')}", fontsize=9)
            ax.set_ylim(0, max(costs) * 1.22)
            ax.set_axisbelow(True)
            if top:
                gains[(model, workload)] = top[1] / tok_s(m8, model, BASE, workload)
                ax.annotate(
                    f"best: {gains[(model, workload)]:.2f}× cheaper",
                    (0.98, 0.97),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                )
            full_gains[(model, workload)] = costs[0] / costs[-1]
        row[0].set_ylabel("$ per 1M output tokens")
    fig.tight_layout()
    large = {w: gains[(models[0], w)] for w in workloads if (models[0], w) in gains}
    top_w, low_w = max(large, key=large.get), min(large, key=large.get)
    worse = sum(full_gains[key] < gains[key] * 0.99 for key in gains)
    caption = (
        f"The best measured stack serves {_name(models[0])} at {large[low_w]:.1f}–{large[top_w]:.1f}× lower "
        f"cost per token than stock BF16 vLLM on the same L4 (most on {_short(top_w)}, least on "
        f"{_short(low_w)}); on {worse} of {len(gains)} model–workload pairs it is not the full stack, "
        "because the last technique added made serving dearer (hatched steps)."
    )
    return fig, caption


# ---- 2. interaction matrix ---------------------------------------------------------------------------------


def interaction_matrix(m8: Records, model: str = LARGE) -> tuple[plt.Figure, str]:
    """Measured combined gain ÷ product of the two gains alone, for every pair and workload."""
    workloads = WORKLOADS[:4]
    grid = np.array([[interaction(m8, model, a, b, w) or np.nan for w in workloads] for a, b in pairs()])
    noise = max((abs(d) for label, _, d in repeat_differences(m8, model) if label == BASE), default=0.0)
    span = max(float(np.nanmax(grid)), 1 / float(np.nanmin(grid)), 1.2)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    image = ax.imshow(grid, cmap="PuOr_r", norm=LogNorm(vmin=1 / span, vmax=span), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                weight = "bold" if abs(grid[i, j] - 1) > 0.05 else "normal"
                dark = abs(np.log(grid[i, j])) > 0.6 * np.log(span)
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight=weight,
                    color="white" if dark else "black",
                )
    ax.set_xticks(range(len(workloads)), [SHORT[w] for w in workloads], fontsize=8)
    ax.set_yticks(range(len(pairs())), [f"{TECHNIQUES[a]} + {TECHNIQUES[b]}" for a, b in pairs()], fontsize=8)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="combined gain ÷ product of the gains alone")
    ax.set_title(
        f"{_name(model)}: do two techniques' gains multiply? (bold: more than 5% from 1)", fontsize=10
    )
    fig.tight_layout()
    flat = [
        (grid[i, j], pairs()[i], workloads[j]) for i in range(grid.shape[0]) for j in range(grid.shape[1])
    ]
    flat = [f for f in flat if not np.isnan(f[0])]
    worst, top = min(flat), max(flat)
    near = sum(abs(value - 1) <= 0.05 for value, _, _ in flat)

    def cell(entry) -> str:
        value, (a, b), workload = entry
        return f"{TECHNIQUES[a]} with {TECHNIQUES[b]} on {_short(workload)} ({value:.2f})"

    caption = (
        f"{near} of {len(flat)} pairs multiply to within 5%. The pair that competes most is {cell(worst)}; "
        f"the pair that helps each other most is {cell(top)}. Two runs of the stock server differ by up to "
        f"{noise:.0%}."
    )
    return fig, caption


# ---- 3. the host's chain -----------------------------------------------------------------------------------

CHAIN_COLORS = {
    "BF16 weights, FlashAttention": OKABE_ITO["sky_blue"],
    "FP8 weights, FlashAttention": SERIES["fp8"][0],
    "FlashInfer + speculation": SERIES["speculative"][0],
    "FlashInfer, no speculation": SERIES["kv_quant"][0],
}


def _chain_kind(label: str) -> str:
    on = letters(label)
    if set(on) & {"k", "f"}:
        return "FlashInfer + speculation" if "s" in on else "FlashInfer, no speculation"
    return "FP8 weights, FlashAttention" if "w" in on else "BF16 weights, FlashAttention"


def host_chain(m8: Records) -> tuple[plt.Figure, str]:
    """Step time at one user: every server on piecewise graphs next to its closest full-graph server."""
    rows = []
    for model in (SMALL, LARGE):
        for label in report.labels_run(m8, model):
            if label.endswith("-r2"):
                continue
            if report.graph_mode(report.graphs_of(m8, model, label)) != "piecewise":
                continue
            twin = next((t for t in report.FULL_GRAPH_TWINS.get(label, []) if step_ms(m8, model, t)), None)
            if twin:
                rows.append((model, label, step_ms(m8, model, label), step_ms(m8, model, twin), twin))
    rows.sort(key=lambda r: (r[0] != SMALL, r[3]))
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    for i, (_, label, piecewise, full, _) in enumerate(rows):
        ax.bar(i - 0.2, full, width=0.38, color=BASELINE_GRAY)
        ax.bar(i + 0.2, piecewise, width=0.38, color=CHAIN_COLORS[_chain_kind(label)])
        ax.annotate(f"{full:.0f}", (i - 0.2, full), ha="center", va="bottom", fontsize=7.5)
        ax.annotate(f"{piecewise:.0f}", (i + 0.2, piecewise), ha="center", va="bottom", fontsize=7.5)
    ticks = [f"{_name(r[0]).replace('Qwen3-', '')}\n{r[1]}\nvs {r[4]}" for r in rows]
    ax.set_xticks(range(len(rows)), ticks, fontsize=7.5)
    split = sum(r[0] == SMALL for r in rows)
    if 0 < split < len(rows):
        ax.axvline(split - 0.5, color=BASELINE_GRAY, linewidth=0.8, linestyle=":")
    handles = [Patch(color=BASELINE_GRAY, label="closest server with a full CUDA graph")]
    handles += [Patch(color=color, label=f"piecewise graphs: {kind}") for kind, color in CHAIN_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
    ax.set(
        ylabel="ms per engine step, one user",
        title="Without a full CUDA graph a step waits for the host",
        ylim=(0, max(r[2] for r in rows) * 1.45),
    )
    ax.set_axisbelow(True)
    fig.tight_layout()
    bound = [r for r in rows if r[2] > 1.15 * r[3]]
    hidden = [r for r in rows if r[2] <= 1.15 * r[3]]
    worst = max(rows, key=lambda r: r[2] / r[3])
    caption = (
        f"{len(bound)} of {len(rows)} servers on piecewise graphs are slower than their full-graph twin, "
        f"by up to {worst[2] / worst[3]:.1f}× (`{worst[1]}` on {_name(worst[0])}: {worst[2]:.0f} ms per step "
        f"against {worst[3]:.0f})."
    )
    if hidden:
        caption += (
            f" The {len(hidden)} that are not are the ones whose GPU already needs "
            f"{min(r[3] for r in hidden):.0f} ms or more per step: the host's time hides behind the GPU's."
        )
    return fig, caption


# ---- 4. recommendation map ---------------------------------------------------------------------------------

CANDIDATES = report.CANDIDATE_STACKS
CANDIDATE_COLOR = {
    "base": BASELINE_GRAY,
    "w": "#9ecae1",
    "wk": "#3182bd",
    "ws": "#fdae6b",
    "wks": "#e6550d",
    "a": "#c7e9c0",
    "ak": "#31a354",
    "as": "#dadaeb",
    "aks": "#756bb1",
}
MEASURED_AT = {  # workload → (users, context) of the map cell closest to it
    "m8_latency": (1, 512),
    "spec_mixed": (64, 512),
    "capacity": (64, 4096),
    "long_32k": (1, 32768),
}


def best_stack(
    cfg: Any,
    hw: Hardware,
    cal: Calibration,
    starts: dict[str, Any],
    model: str,
    users: int,
    context: int,
    kept: float,
    draft_bytes: float,
    allowed: list[str],
) -> tuple[str, float]:
    """The candidate the serving model expects to give the most tokens/s for this many users and context."""
    predictions = report.predict_candidates(cfg, hw, cal, starts, model, users, context, kept, draft_bytes)
    scores = {label: predictions[label]["tok_s"] for label in allowed}
    winner = max(scores, key=scores.get)
    return winner, scores[winner] / predictions[BASE]["tok_s"]


def recommendation_map(
    m8: Records,
    cfg: Any,
    hw: Hardware,
    cal: Calibration,
    starts: dict[str, Any],
    model: str,
    kept: float,
    draft_bytes: float,
) -> tuple[plt.Figure, str]:
    """Which configuration the model recommends over (concurrency × context), with and without INT4.

    `cal` is the M8-informed calibration: it knows that FP8 KV with speculation waits for the host. A ★ marks
    the cells next to a measured workload where the model's pick is also the fastest measured stack.
    """
    users = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    panels = [
        ("FP8-class quality only", [c for c in CANDIDATES if not c.startswith("a")]),
        ("INT4 weights allowed (lower quality: M4)", list(CANDIDATES)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), sharey=True)
    counts: dict[str, int] = {}
    agree, checked = 0, 0
    for ax, (title, allowed) in zip(axes, panels, strict=True):
        picks = {}
        for i, n in enumerate(users):
            for j, context in enumerate(contexts):
                label, gain = best_stack(cfg, hw, cal, starts, model, n, context, kept, draft_bytes, allowed)
                picks[(n, context)] = label
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=CANDIDATE_COLOR[label]))
                ax.text(j + 0.5, i + 0.6, label, ha="center", va="center", fontsize=8, fontweight="bold")
                ax.text(j + 0.5, i + 0.28, f"{gain:.1f}×", ha="center", va="center", fontsize=7)
                if ax is axes[0]:
                    counts[label] = counts.get(label, 0) + 1
        if ax is axes[0]:
            for workload, cell in MEASURED_AT.items():
                top = best(m8, model, workload, allow_int4=False)
                if not top:
                    continue
                checked += 1
                measured = top[0].replace("p", "") or BASE  # the map has no prefix caching: it never hurts
                if measured == picks[cell]:
                    agree += 1
                    i, j = users.index(cell[0]), contexts.index(cell[1])
                    ax.text(j + 0.86, i + 0.8, "★", ha="center", va="center", fontsize=9)
        ax.set_xlim(0, len(contexts))
        ax.set_ylim(0, len(users))
        ax.set_xticks(np.arange(len(contexts)) + 0.5, [f"{c:,}" for c in contexts], fontsize=8)
        ax.set_yticks(np.arange(len(users)) + 0.5, [str(n) for n in users], fontsize=8)
        ax.set(xlabel="context per request (tokens)", title=title)
        ax.grid(False)
    axes[0].set_ylabel("concurrent users")
    handles = [Patch(color=CANDIDATE_COLOR[c], label=_candidate_name(c)) for c in CANDIDATES]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        f"{_name(model)} on an L4: the configuration the M8-informed model expects to be fastest "
        "(256-token answers; × = tokens/s over stock BF16; ★ = also the fastest measured)",
        fontsize=10,
    )
    fig.tight_layout()
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    caption = (
        f"The M8-informed model's recommendation for {_name(model)}, FP8-class quality only: "
        + ", ".join(f"`{label}` in {count / total:.0%} of the cells" for label, count in ranked[:3])
        + f". It names the fastest measured stack on {agree} of the {checked} measured workloads (★); "
        "everywhere else the map is a prediction."
    )
    return fig, caption


def _candidate_name(label: str) -> str:
    weights, fp8_kv, spec = CANDIDATES[label]
    parts = [{"bf16": "BF16", "fp8": "FP8 weights", "int4": "INT4 weights"}[weights]]
    parts += ["FP8 KV"] if fp8_kv else []
    parts += ["speculation"] if spec else []
    return f"{label}: " + " + ".join(parts)


# ---- 5. quality against cost -------------------------------------------------------------------------------


def quality_cost(
    m8: Records, perplexity: dict[str, float], dollars_per_hour: float, model: str = LARGE
) -> tuple[plt.Figure, str]:
    """Every deployable configuration: cost on the busy workload against the perplexity of its lossy parts.

    `perplexity` maps a lossy set ("", "w", "k", "wk", "a") to the perplexity measured for it in vLLM.
    Prefix caching and speculation do not change it, so a configuration takes its lossy set's value.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    reference = perplexity[""]
    points = []
    for label in report.candidates(m8, model):
        lossy = "".join(letter for letter in letters(label) if letter in "wak")
        rate = tok_s(m8, model, label, "spec_mixed")
        if lossy in perplexity and rate:
            points.append((dollars(rate, dollars_per_hour), 100 * (perplexity[lossy] / reference - 1), label))
    for cost, change, label in points:
        marker = "X" if "s" in letters(label) else "o"
        ax.scatter(cost, change, color=_lossy_color(label), marker=marker, s=46, zorder=3)
        ax.annotate(label, (cost, change), xytext=(4, 4), textcoords="offset points", fontsize=7)
    frontier = _pareto(points)
    ax.plot([c for c, _, _ in frontier], [q for _, q, _ in frontier], "--", color=BASELINE_GRAY, linewidth=1)
    ax.axhline(0, color=BASELINE_GRAY, linewidth=0.8)
    ax.set(
        xlabel="$ per 1M output tokens, busy workload (64 users)",
        ylabel="perplexity vs BF16 (%)",
        title=f"{_name(model)}: quality against cost, every measured configuration",
    )
    handles = [
        Patch(color=BASELINE_GRAY, label="BF16 weights and cache"),
        Patch(color=SERIES["fp8"][0], label="FP8 weights"),
        Patch(color=SERIES["kv_quant"][0], label="FP8 KV cache"),
        Patch(color=OKABE_ITO["black"], label="FP8 weights + FP8 KV"),
        Patch(color=SERIES["int4"][0], label="INT4 weights"),
    ]
    ax.legend(handles=handles, fontsize=8, title="× marks: with speculation", title_fontsize=8)
    fig.tight_layout()
    cheapest = min(points)
    base = next(p for p in points if p[2] == BASE)
    fp8 = min(p for p in points if "a" not in letters(p[2]))
    caption = (
        f"The cheapest measured configuration on the busy workload, `{cheapest[2]}`, costs "
        f"{base[0] / cheapest[0]:.2f}× less than stock BF16 for a perplexity change of {cheapest[1]:+.1f}%; "
        f"without INT4 weights the cheapest is `{fp8[2]}` at {base[0] / fp8[0]:.2f}× and {fp8[1]:+.1f}%. "
        f"{len(frontier)} of {len(points)} configurations are on the frontier."
    )
    return fig, caption


def _lossy_color(label: str) -> str:
    on = letters(label)
    if "a" in on:
        return SERIES["int4"][0]
    if "w" in on and "k" in on:
        return OKABE_ITO["black"]
    if "w" in on:
        return SERIES["fp8"][0]
    return SERIES["kv_quant"][0] if "k" in on else BASELINE_GRAY


def _pareto(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Configurations no other one beats on both cost and quality, by cost."""
    frontier, best_quality = [], float("inf")
    for point in sorted(points):
        if point[1] < best_quality:
            frontier.append(point)
            best_quality = point[1]
    return frontier


# ---- 6. predicted against measured -------------------------------------------------------------------------

GROUPS = ["no speculation", "speculation", "FP8 KV + speculation"]


def _group(label: str) -> str:
    on = letters(label)
    if "s" in on and "k" in on:
        return "FP8 KV + speculation"
    return "speculation" if "s" in on else "no speculation"


def predicted_vs_measured(m8: Records, frozen: dict[str, Any]) -> tuple[plt.Figure, str]:
    """The predictions frozen before M8 against what M8 measured, with a ±15% band."""
    rows = prediction_errors(m8, frozen)
    colors = dict(
        zip(GROUPS, [OKABE_ITO["black"], SERIES["speculative"][0], SERIES["speculative"][0]], strict=True)
    )
    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    lo = min(min(r["measured"], r["tok_s"]) for r in rows) * 0.7
    hi = max(max(r["measured"], r["tok_s"]) for r in rows) * 1.4
    line = np.array([lo, hi])
    ax.fill_between(line, line * 0.85, line * 1.15, color=BASELINE_GRAY, alpha=0.18)
    ax.plot(line, line, color=BASELINE_GRAY, linewidth=1)
    for workload in WORKLOADS:
        for group in GROUPS:
            points = [r for r in rows if r["workload"] == workload and _group(r["label"]) == group]
            if points:
                hollow = group == "FP8 KV + speculation"
                ax.scatter(
                    [r["measured"] for r in points],
                    [r["tok_s"] for r in points],
                    marker=MARKERS[workload],
                    s=38,
                    edgecolors=colors[group],
                    facecolors="white" if hollow else colors[group],
                    linewidths=1.2,
                    zorder=3,
                )
    handles = [Patch(color=BASELINE_GRAY, alpha=0.3, label="±15%")]
    handles += [Patch(color=colors["no speculation"], label="no speculation")]
    handles += [Patch(color=colors["speculation"], label="speculation")]
    handles += [
        Patch(facecolor="white", edgecolor=colors["speculation"], label="FP8 KV + speculation (hollow)")
    ]
    handles += [
        plt.Line2D([], [], marker=MARKERS[w], color="black", linestyle="", label=SHORT[w].split(chr(10))[0])
        for w in WORKLOADS
    ]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="measured (output tokens/s)",
        ylabel="predicted before the run (output tokens/s)",
        title="The serving model against M8: every planned server and workload",
        xlim=(lo, hi),
        ylim=(lo, hi),
    )
    fig.tight_layout()

    def median(group: str) -> float:
        errors = [abs(r["error"]) for r in rows if _group(r["label"]) == group]
        return statistics.median(errors) if errors else float("nan")

    errors = [r["error"] for r in rows]
    within = sum(abs(e) <= 0.15 for e in errors) / len(errors)
    caption = (
        f"Frozen before any M8 server ran, the model's {len(errors)} predictions have a median error of "
        f"{statistics.median(abs(e) for e in errors):.0%} ({within:.0%} within 15%): "
        f"{median('no speculation'):.0%} without speculation, {median('speculation'):.0%} with it, and "
        f"{median('FP8 KV + speculation'):.0%} where FP8 KV meets speculation, which the model had no term "
        "for."
    )
    return fig, caption


# ---- 7. leave one out --------------------------------------------------------------------------------------


def leave_one_out(m8: Records, model: str = LARGE) -> tuple[plt.Figure, str]:
    """What the full stack loses when one technique is removed, next to what that technique gives alone."""
    workloads = WORKLOADS[:4]
    fig, axes = plt.subplots(1, len(workloads), figsize=(3.4 * len(workloads), 4.0), sharey=True)
    hurts, cases = [], 0
    for ax, workload in zip(axes, workloads, strict=True):
        for i, letter in enumerate(TECHNIQUES):
            alone = speedup(m8, model, letter, workload)
            stacked = speedup(m8, model, FULL, workload, over=without(FULL, letter))
            if alone is None or stacked is None:
                continue
            cases += 1
            ax.bar(i - 0.2, alone, width=0.38, color=STEP_COLOR[letter], alpha=0.45)
            ax.bar(i + 0.2, stacked, width=0.38, color=STEP_COLOR[letter])
            ax.annotate(f"{stacked:.2f}×", (i + 0.2, stacked), ha="center", va="bottom", fontsize=7.5)
            if stacked < 0.97:
                hurts.append((stacked, letter, workload, alone))
        ax.axhline(1, color=BASELINE_GRAY, linewidth=0.8)
        ticks = ["FP8\nweights", "FP8\nKV", "prefix\ncache", "spec.\ndecode"]
        ax.set_xticks(range(len(TECHNIQUES)), ticks, fontsize=7.5)
        ax.set_title(SHORT[workload].replace("\n", ", "), fontsize=9)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("tokens/s with the technique ÷ without it")
    handles = [
        Patch(color=BASELINE_GRAY, alpha=0.45, label="alone (on stock BF16)"),
        Patch(color=BASELINE_GRAY, label="in the full stack"),
    ]
    axes[-1].legend(handles=handles, fontsize=8)
    fig.suptitle(
        f"{_name(model)}: what each technique is worth alone, and inside the full stack", fontsize=10
    )
    fig.tight_layout()
    if hurts:
        value, letter, workload, alone = min(hurts)
        caption = (
            f"In {len(hurts)} of {cases} cases the full stack is faster *without* a technique. The largest: "
            f"removing {TECHNIQUES[letter]} on {_short(workload)} makes the stack {1 / value:.2f}× faster, "
            f"although alone it gives {alone:.2f}×."
        )
    else:
        caption = "Every technique still pays inside the full stack on every workload."
    return fig, caption


def make_all(
    m8: Records,
    frozen: dict[str, Any],
    configs: dict[str, Any],
    hw: Hardware,
    dollars_per_hour: float,
    perplexity: dict[str, dict[str, float]],
    head: dict[str, Any],
    projected: dict[str, float] | None,
    out_dir: str | Path,
) -> list[Path]:
    apply_style()
    cal = report.informed_calibration(report.calibration_of(frozen), m8)
    kept = frozen["expected_loads"]["spec_mixed"]["tokens_per_pass"]
    figures = {
        "m8_waterfall": lambda: final_waterfall(m8, dollars_per_hour, projected),
        "m8_interactions": lambda: interaction_matrix(m8),
        "m8_host_chain": lambda: host_chain(m8),
        "m8_recommendation": lambda: recommendation_map(
            m8,
            configs[LARGE],
            hw,
            cal,
            frozen["startup_memory"],
            LARGE,
            kept,
            report.eagle_bytes(configs[LARGE], head),
        ),
        "m8_quality_cost": lambda: quality_cost(m8, perplexity[LARGE], dollars_per_hour),
        "m8_predicted": lambda: predicted_vs_measured(m8, frozen),
        "m8_leave_one_out": lambda: leave_one_out(m8),
    }
    written: list[Path] = []
    for name, build in figures.items():
        try:
            fig, caption = build()
        except (KeyError, TypeError, ValueError, ZeroDivisionError, StopIteration, IndexError) as err:
            print(f"skipping {name}: {type(err).__name__}: {err}")
            continue
        written += save_figure(fig, name, caption, out_dir)
    return written
