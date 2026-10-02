"""M8 figures: the full stack. Each returns (figure, caption computed from the data)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from fastserve.perfmodel.serving import FLASHINFER, Calibration, Hardware, Load, Speculation, Stack, predict
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
    dollars,
    interaction,
    pairs,
    prediction_errors,
    repeat_differences,
    speedup,
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

    `projected` maps a model to the tokens/s M7's kernel projection gives for the long workload: drawn
    hatched, because it was never run in vLLM.
    """
    models = [m for m in (LARGE, SMALL) if tok_s(m8, m, BASE, "m8_latency")]
    workloads = [w for w in WORKLOADS if tok_s(m8, models[0], BASE, w)]
    fig, axes = plt.subplots(len(models), len(workloads), figsize=(3.3 * len(workloads), 3.9 * len(models)))
    axes = np.atleast_2d(axes)
    best = {}
    for row, model in zip(axes, models, strict=True):
        for ax, workload in zip(row, workloads, strict=True):
            costs = [dollars(tok_s(m8, model, label, workload), dollars_per_hour) for label in LADDER]
            if None in costs:
                ax.axis("off")
                continue
            ax.bar(0, costs[0], color=BASELINE_GRAY)
            for i, letter in enumerate(TECHNIQUES, start=1):
                lo, hi = sorted((costs[i - 1], costs[i]))
                ax.bar(i, hi - lo, bottom=lo, color=STEP_COLOR[letter])
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
            extra = (projected or {}).get(model) if workload == "long_32k" else None
            if extra:
                cost = dollars(extra, dollars_per_hour)
                ax.bar(len(LADDER) + 1, cost, color="white", edgecolor=OKABE_ITO["black"], hatch="///")
                ticks.append("+ kernel 2\n(projected)")
            ax.set_xticks(range(len(ticks)), ticks, fontsize=6.5)
            ax.set_title(f"{_name(model)}, {SHORT[workload].replace(chr(10), ', ')}", fontsize=9)
            ax.set_ylim(0, costs[0] * 1.18)
            ax.annotate(
                f"{costs[0] / costs[-1]:.1f}× cheaper",
                (len(LADDER), costs[-1]),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
            ax.set_axisbelow(True)
            best[(model, workload)] = costs[0] / costs[-1]
        row[0].set_ylabel("$ per 1M output tokens")
    fig.tight_layout()
    large = {w: best[(models[0], w)] for w in workloads if (models[0], w) in best}
    top, low = max(large, key=large.get), min(large, key=large.get)
    caption = (
        "The full stack (FP8 weights, FP8 KV, prefix caching, an EAGLE-3 drafter) serves "
        f"{_name(models[0])} at {large[low]:.1f}–{large[top]:.1f}× lower cost per token than stock BF16 vLLM "
        f"on the same L4: most on {_short(top)}, least on {_short(low)}; each technique pays on the workload "
        "whose bottleneck it removes."
    )
    return fig, caption


# ---- 2. interaction matrix ---------------------------------------------------------------------------------


def interaction_matrix(m8: Records, model: str = LARGE) -> tuple[plt.Figure, str]:
    """Measured combined gain ÷ product of the two gains alone, for every pair and workload."""
    workloads = WORKLOADS[:4]
    grid = np.array([[interaction(m8, model, a, b, w) or np.nan for w in workloads] for a, b in pairs()])
    noise = max((abs(d) for _, _, d in repeat_differences(m8, model)), default=0.0)
    span = max(float(np.nanmax(grid)), 1 / float(np.nanmin(grid)), 1.2)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    image = ax.imshow(grid, cmap="PuOr_r", norm=LogNorm(vmin=1 / span, vmax=span), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                weight = "bold" if abs(grid[i, j] - 1) > 2 * noise else "normal"  # two runs' noise, both ways
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
        f"{_name(model)}: do two techniques' gains multiply? (bold: beyond run-to-run noise)", fontsize=10
    )
    fig.tight_layout()
    flat = [
        (grid[i, j], pairs()[i], workloads[j]) for i in range(grid.shape[0]) for j in range(grid.shape[1])
    ]
    flat = [f for f in flat if not np.isnan(f[0])]
    worst, best = min(flat), max(flat)

    def cell(entry) -> str:
        value, (a, b), workload = entry
        return f"{TECHNIQUES[a]} with {TECHNIQUES[b]} on {_short(workload)} ({value:.2f})"

    caption = (
        "Gains multiply (1.00) only when techniques shorten different parts of a step: the pair that "
        f"competes most is {cell(worst)}, the pair that helps each other most is {cell(best)}; two runs of "
        f"the same server differ by up to {noise:.0%}."
    )
    return fig, caption


# ---- 3. recommendation map ---------------------------------------------------------------------------------

CANDIDATES = {  # label → (weights, FP8 KV, speculation)
    "base": ("bf16", False, False),
    "w": ("fp8", False, False),
    "wk": ("fp8", True, False),
    "ws": ("fp8", False, True),
    "wks": ("fp8", True, True),
    "a": ("int4", False, False),
    "ak": ("int4", True, False),
    "as": ("int4", False, True),
    "aks": ("int4", True, True),
}
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
    output_len: int = 256,
) -> tuple[str, float]:
    """The candidate the serving model expects to give the most tokens/s for this many users and context."""
    scores = {}
    for label in allowed:
        weights, fp8_kv, spec = CANDIDATES[label]
        stack = Stack(
            weights=weights,
            kv="fp8" if fp8_kv else "bf16",
            backend=FLASHINFER if fp8_kv else "flash_attn",
            speculation=Speculation(3, kept, draft_bytes) if spec else None,
        )
        load = Load(users=users, prompt_len=context, output_len=output_len)
        load = replace(load, kv_tokens=report.kv_capacity(model, stack, cfg, starts))
        scores[label] = predict(cfg, hw, stack, cal, load)["tok_s"]
    winner = max(scores, key=scores.get)
    return winner, scores[winner] / scores["base"]


def recommendation_map(
    cfg: Any,
    hw: Hardware,
    cal: Calibration,
    starts: dict[str, Any],
    model: str,
    kept: float,
    draft_bytes: float,
) -> tuple[plt.Figure, str]:
    """Which configuration the model recommends over (concurrency × context), with and without INT4."""
    users = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    panels = [
        ("FP8-class quality only", [c for c in CANDIDATES if not c.startswith("a")]),
        ("INT4 weights allowed (lower quality: M4)", list(CANDIDATES)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), sharey=True)
    counts: dict[str, int] = {}
    for ax, (title, allowed) in zip(axes, panels, strict=True):
        for i, n in enumerate(users):
            for j, context in enumerate(contexts):
                label, gain = best_stack(cfg, hw, cal, starts, model, n, context, kept, draft_bytes, allowed)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=CANDIDATE_COLOR[label]))
                ax.text(j + 0.5, i + 0.6, label, ha="center", va="center", fontsize=8, fontweight="bold")
                ax.text(j + 0.5, i + 0.28, f"{gain:.1f}×", ha="center", va="center", fontsize=7)
                if ax is axes[0]:
                    counts[label] = counts.get(label, 0) + 1
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
        f"{_name(model)} on an L4: the configuration the serving model expects to be fastest "
        "(256-token answers; × = tokens/s over stock BF16)",
        fontsize=10,
    )
    fig.tight_layout()
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    caption = (
        f"The serving model's recommendation for {_name(model)}, FP8-class quality only: "
        + ", ".join(f"`{label}` in {count / total:.0%} of the cells" for label, count in ranked[:3])
        + ": speculation where there is idle compute, FP8 KV where the cache is the step. This map is "
        "predicted, not measured."
    )
    return fig, caption


def _candidate_name(label: str) -> str:
    weights, fp8_kv, spec = CANDIDATES[label]
    parts = [{"bf16": "BF16", "fp8": "FP8 weights", "int4": "INT4 weights"}[weights]]
    parts += ["FP8 KV"] if fp8_kv else []
    parts += ["speculation"] if spec else []
    return f"{label}: " + " + ".join(parts)


# ---- 4. quality against cost -------------------------------------------------------------------------------


def quality_cost(
    m8: Records, perplexity: dict[str, float], dollars_per_hour: float, model: str = LARGE
) -> tuple[plt.Figure, str]:
    """Every measured configuration: cost on the busy workload against the perplexity of its lossy parts.

    `perplexity` maps a lossy set ("", "w", "k", "wk", "a") to the perplexity measured for it in vLLM.
    Prefix caching and speculation do not change it, so a configuration takes its lossy set's value.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    reference = perplexity[""]
    points = []
    for label in report.labels_run(m8, model):
        if label.endswith("-r2"):
            continue
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
    caption = (
        f"The cheapest measured configuration on the busy workload, `{cheapest[2]}`, costs "
        f"{base[0] / cheapest[0]:.1f}× less than stock BF16 for a perplexity change of {cheapest[1]:+.2f}%; "
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


# ---- 5. predicted against measured -------------------------------------------------------------------------


def predicted_vs_measured(m8: Records, frozen: dict[str, Any]) -> tuple[plt.Figure, str]:
    """The predictions frozen before M8 against what M8 measured, with a ±15% band."""
    rows = prediction_errors(m8, frozen)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    lo = min(min(r["measured"], r["tok_s"]) for r in rows) * 0.7
    hi = max(max(r["measured"], r["tok_s"]) for r in rows) * 1.4
    line = np.array([lo, hi])
    ax.fill_between(line, line * 0.85, line * 1.15, color=BASELINE_GRAY, alpha=0.18, label="±15%")
    ax.plot(line, line, color=BASELINE_GRAY, linewidth=1)
    for workload in WORKLOADS:
        for spec, face in ((False, None), (True, "white")):
            group = [r for r in rows if r["workload"] == workload and ("s" in letters(r["label"])) == spec]
            if group:
                color = OKABE_ITO["vermillion"] if spec else OKABE_ITO["blue"]
                ax.scatter(
                    [r["measured"] for r in group],
                    [r["tok_s"] for r in group],
                    marker=MARKERS[workload],
                    s=38,
                    edgecolors=color,
                    facecolors=face or color,
                    linewidths=1.2,
                    label=f"{SHORT[workload].split(chr(10))[0]}{', with speculation' if spec else ''}",
                    zorder=3,
                )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="measured (output tokens/s)",
        ylabel="predicted before the run (output tokens/s)",
        title="The serving model against M8: every server and workload",
        xlim=(lo, hi),
        ylim=(lo, hi),
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    errors = [r["error"] for r in rows]
    plain = [r["error"] for r in rows if "s" not in letters(r["label"])]
    spec = [r["error"] for r in rows if "s" in letters(r["label"])]
    within = sum(abs(e) <= 0.15 for e in errors) / len(errors)
    caption = (
        f"Frozen before any M8 server ran, the model's {len(errors)} predictions have a median error of "
        f"{np.median(np.abs(errors)):.0%} ({within:.0%} within 15%): {np.median(np.abs(plain)):.0%} without "
        f"speculation and {np.median(np.abs(spec)) if spec else float('nan'):.0%} with it."
    )
    return fig, caption


# ---- 6. leave one out --------------------------------------------------------------------------------------


def leave_one_out(m8: Records, model: str = LARGE) -> tuple[plt.Figure, str]:
    """What the full stack loses when one technique is removed, next to what that technique gives alone."""
    workloads = WORKLOADS[:4]
    fig, axes = plt.subplots(1, len(workloads), figsize=(3.4 * len(workloads), 4.0), sharey=True)
    largest = (0.0, "", "")
    for ax, workload in zip(axes, workloads, strict=True):
        for i, letter in enumerate(TECHNIQUES):
            alone = speedup(m8, model, letter, workload)
            stacked = speedup(m8, model, FULL, workload, over=without(FULL, letter))
            if alone is None or stacked is None:
                continue
            ax.bar(i - 0.2, alone, width=0.38, color=STEP_COLOR[letter], alpha=0.45)
            ax.bar(i + 0.2, stacked, width=0.38, color=STEP_COLOR[letter])
            ax.annotate(f"{stacked:.2f}×", (i + 0.2, stacked), ha="center", va="bottom", fontsize=7.5)
            if stacked > largest[0]:
                largest = (stacked, letter, workload)
        ax.axhline(1, color=BASELINE_GRAY, linewidth=0.8)
        ax.set_xticks(
            range(len(TECHNIQUES)),
            ["FP8\nweights", "FP8\nKV", "prefix\ncache", "spec.\ndecode"],
            fontsize=7.5,
        )
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
    value, letter, workload = largest
    alone = speedup(m8, model, letter, workload)
    caption = (
        f"The technique the full stack can least afford to lose is {TECHNIQUES[letter]} on "
        f"{_short(workload)}: {value:.2f}× with it, against {alone:.2f}× when added to stock BF16 alone. A "
        "technique's gain alone and its gain in the stack differ wherever techniques interact."
    )
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
    cal = report.calibration_of(frozen)
    kept = frozen["expected_loads"]["spec_mixed"]["tokens_per_pass"]
    figures = {
        "m8_waterfall": lambda: final_waterfall(m8, dollars_per_hour, projected),
        "m8_interactions": lambda: interaction_matrix(m8),
        "m8_recommendation": lambda: recommendation_map(
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
