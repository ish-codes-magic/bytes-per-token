"""M6 figures: speculative decoding. Each returns (figure, caption computed from the data)."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fastserve.report.m6 import (
    DRAFTER_LABELS,
    ONE_USER,
    TASK_LABELS,
    TASKS,
    USERS,
    _mean,
    _newest,
    agreement,
    one_user_speedup,
    parse_label,
    server_label,
    speedup,
    tok_s,
    tokens_per_pass,
    vllm_acceptance,
)
from fastserve.spec.simulate import expected_tokens
from fastserve.viz.style import BASELINE_GRAY, OKABE_ITO, apply_style, save_figure

Records = list[dict[str, Any]]
ACCEPTED, WRITTEN = (
    "#BFE3F7",
    "#F9D9A6",
)  # light sky blue: drafted and accepted; light orange: the target's own
METHOD_STYLE = {  # one color per method everywhere; draft length changes the line style
    "draft": (OKABE_ITO["vermillion"], "o"),
    "draftawq": (OKABE_ITO["orange"], "v"),
    "ngram": (OKABE_ITO["green"], "s"),
    "eagle3": (OKABE_ITO["blue"], "D"),
}
K_STYLE = {1: ":", 3: "-", 5: "--", 6: "--"}
SHORT = {"draft": "0.6B drafter", "draftawq": "INT4 drafter", "eagle3": "EAGLE-3", "ngram": "n-gram"}
TASK_STYLE = {
    "chat": (OKABE_ITO["purple"], "o"),
    "code": (OKABE_ITO["blue"], "s"),
    "math": (OKABE_ITO["orange"], "^"),
    "summarize": (OKABE_ITO["green"], "D"),
}


def _users(n: int) -> str:
    return "1 user" if n == 1 else f"{n} users"


def _spec_labels(m6: Records) -> list[str]:
    labels = {m["server"] for m in _newest(m6, "serving") if m["server"].startswith("bf16-")}
    labels = [x for x in labels if parse_label(x)[1] != "none"]
    return sorted(labels, key=lambda x: (parse_label(x)[1], parse_label(x)[2] or 0))


# ---- the token highlighter --------------------------------------------------------------------------------


def _draw_tokens(
    ax: plt.Axes, pieces: list[str], flags: list[bool], width: int = 104, max_lines: int = 9
) -> None:
    """Text as colored token boxes: monospace, wrapped at `width` characters."""
    ax.set(xlim=(0, width), ylim=(max_lines, 0))
    ax.axis("off")
    col = line = 0
    for piece, drafted in zip(pieces, flags, strict=True):
        text = piece.replace("\n", "↵").replace("\t", "  ").replace("$", r"\$")
        if col + len(text) > width or (col and piece.startswith("\n")):
            col, line = 0, line + 1
        if line >= max_lines:
            break
        ax.text(
            col,
            line + 0.5,
            text,
            family="monospace",
            fontsize=7,
            va="center",
            fontweight="normal" if drafted else "bold",  # not color alone: the target's own tokens are bold
            bbox={"facecolor": ACCEPTED if drafted else WRITTEN, "edgecolor": "none", "pad": 1.2},
        )
        col += len(text)


def highlight(m6: Records) -> tuple[plt.Figure, str]:
    """Which tokens the drafter got right: one answer per task, for the model drafter and n-gram lookup."""
    shown = {(m["drafter"], m["task"]): m for m in _newest(m6, "m6_highlight") if m["target"] == "bf16"}
    drafters = [d for d in DRAFTER_LABELS if any(k[0] == d for k in shown)]
    fig, axes = plt.subplots(
        len(TASKS), len(drafters), figsize=(6.5 * len(drafters), 1.9 * len(TASKS)), squeeze=False
    )
    shares = {}
    for col, drafter in enumerate(drafters):
        for row, task in enumerate(TASKS):
            m = shown.get((drafter, task))
            if not m:
                axes[row, col].axis("off")
                continue
            _draw_tokens(axes[row, col], m["pieces"], m["from_draft"])
            share = sum(m["from_draft"]) / len(m["from_draft"])
            shares[drafter, task] = share
            axes[row, col].set_title(
                f"{TASK_LABELS[task]}, {DRAFTER_LABELS[drafter]} (k = {m['k']}): {share:.0%} drafted",
                fontsize=9,
                loc="left",
            )
    fig.tight_layout()
    best = max(shares, key=shares.get) if shares else None
    worst = min(shares, key=shares.get) if shares else None
    caption = "Blue tokens were drafted and accepted; bold orange ones the target wrote itself"
    if best:
        top, low = TASK_LABELS[best[1]].lower(), TASK_LABELS[worst[1]].lower()
        caption += (
            f". {DRAFTER_LABELS[best[0]]} drafted {shares[best]:.0%} of the {top} answer; "
            f"{DRAFTER_LABELS[worst[0]]} drafted {shares[worst]:.0%} of the {low} answer"
        )
    return fig, caption + "."


def highlight_html(m6: Records) -> str:
    """The same as a standalone page with the whole outputs (for the dashboard)."""
    blocks = []
    for m in _newest(m6, "m6_highlight"):
        if m["target"] != "bf16":
            continue
        spans = "".join(
            f'<span class="{"d" if drafted else "t"}">{html.escape(piece)}</span>'
            for piece, drafted in zip(m["pieces"], m["from_draft"], strict=True)
        )
        share = sum(m["from_draft"]) / len(m["from_draft"])
        title = f"{TASK_LABELS[m['task']]}, {DRAFTER_LABELS[m['drafter']]}, k = {m['k']}: {share:.0%} drafted"
        blocks.append(f"<h3>{html.escape(title)}</h3><pre>{spans}</pre>")
    style = (
        "body{font-family:sans-serif;max-width:60rem;margin:2rem auto}"
        "pre{white-space:pre-wrap;line-height:1.7}"
        f".d{{background:{ACCEPTED}}}.t{{background:{WRITTEN};font-weight:bold}}"
    )
    legend = (
        '<p><span class="d">drafted and accepted</span> &nbsp; '
        '<span class="t">written by the target</span></p>'
    )
    head = f"<!doctype html><meta charset='utf-8'><title>Token acceptance</title><style>{style}</style>"
    return f"{head}<h1>Which tokens did the drafter get right?</h1>{legend}{''.join(blocks)}"


# ---- reference results ------------------------------------------------------------------------------------


def accepted_lengths(m6: Records, k: int = 8) -> tuple[plt.Figure, str]:
    """How many draft tokens a round accepts, per drafter and task, with k draft tokens offered."""
    drafters = [d for d in DRAFTER_LABELS if agreement(m6, "bf16", d)]
    fig, axes = plt.subplots(
        len(drafters), len(TASKS), figsize=(12, 2.6 * len(drafters)), sharey=True, squeeze=False
    )
    zero = {}
    for row, drafter in enumerate(drafters):
        for col, task in enumerate(TASKS):
            hist = (
                (agreement(m6, "bf16", drafter, task) or {})
                .get("by_k", {})
                .get(str(k), {})
                .get("accepted_histogram")
            )
            ax = axes[row, col]
            if not hist:
                ax.axis("off")
                continue
            share = np.array(hist + [0] * (k + 1 - len(hist))) / sum(hist)
            color = METHOD_STYLE[
                "ngram" if drafter == "ngram" else "draftawq" if "awq" in drafter else "draft"
            ][0]
            ax.bar(range(k + 1), share, color=color)
            ax.set_title(f"{TASK_LABELS[task]}, {DRAFTER_LABELS[drafter]}", fontsize=9)
            zero[drafter, task] = share[0]
            if row == len(drafters) - 1:
                ax.set_xlabel(f"draft tokens accepted (of {k})")
        axes[row, 0].set_ylabel("share of rounds")
    fig.tight_layout()
    caption = "Acceptance is all-or-nothing more often than a coin flip per token would give"
    if ("small", "chat") in zero and ("ngram", "chat") in zero:
        caption = (
            f"With {k} tokens drafted, a round of the small model accepts none on chat "
            f"{zero['small', 'chat']:.0%} of the time; n-gram lookup accepts none "
            f"{zero['ngram', 'chat']:.0%} of the time there, because it "
            "rarely has anything to propose."
        )
    return fig, caption


def theory_vs_measured(m6: Records) -> tuple[plt.Figure, str]:
    """E(α, k) for independent acceptances, against the replayed and vLLM-measured tokens per target pass."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = np.arange(1, 9)
    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9):
        ax.plot(ks, [expected_tokens(alpha, int(k)) for k in ks], color=BASELINE_GRAY, lw=0.8)
        ax.annotate(
            f"α = {alpha}",
            (8, expected_tokens(alpha, 8)),
            fontsize=7,
            va="center",
            xytext=(4, 0),
            textcoords="offset points",
        )
    for task in TASKS:
        color, marker = TASK_STYLE[task]
        points = [(k, tokens_per_pass(m6, "bf16", "small", task, int(k))) for k in ks]
        points = [(k, e) for k, e in points if e]
        if points:
            ax.plot(
                *zip(*points, strict=True),
                color=color,
                marker=marker,
                ls="",
                label=f"replay, {TASK_LABELS[task].lower()}",
            )
    row = agreement(m6, "bf16", "small")
    gap = None
    if row and "rates" in row:
        alpha = row["rates"]["agree"]
        ax.plot(
            ks,
            [expected_tokens(alpha, int(k)) for k in ks],
            color="black",
            ls="--",
            label=f"independent, α = {alpha:.2f} (all tasks)",
        )
        measured = tokens_per_pass(m6, "bf16", "small", "all", 8)
        gap = measured / expected_tokens(alpha, 8) - 1 if measured else None
    vllm = [(k, vllm_acceptance(m6, f"bf16-draft-k{k}", ONE_USER)) for k in (1, 3, 5)]
    vllm = [(k, v["tokens_per_round"]) for k, v in vllm if v]
    if vllm:
        ax.plot(
            *zip(*vllm, strict=True),
            color="black",
            marker="*",
            ms=12,
            ls="",
            label="vLLM counters, all tasks",
        )
    ax.set(xlabel="draft length k", ylabel="tokens per target pass", xlim=(0.5, 9.3))
    ax.legend(fontsize=7)
    caption = "Measured tokens per target pass against the independent-acceptance formula"
    if gap is not None:
        caption += (
            f": at k = 8 the replay yields {gap:+.0%} versus the formula at the measured agreement. "
            "Agreements aren't independent: every round starts right after a miss, where the drafter is "
            "least reliable"
        )
    return fig, caption + "."


# ---- vLLM -------------------------------------------------------------------------------------------------


def speedup_vs_users(m6: Records) -> tuple[plt.Figure, str]:
    """Speedup over no speculation as users are added: one user (mean of tasks), then mixed tasks."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = [1, *USERS]
    curves = {}
    for label in _spec_labels(m6):
        _, method, k = parse_label(label)
        ys = [one_user_speedup(m6, label), *(speedup(m6, label, "spec_mixed", u) for u in USERS)]
        if not any(ys):
            continue
        color, marker = METHOD_STYLE.get(method, (BASELINE_GRAY, "o"))
        points = [(x, y) for x, y in zip(xs, ys, strict=True) if y]
        ax.plot(
            *zip(*points, strict=True),
            color=color,
            marker=marker,
            ls=K_STYLE.get(k, "-"),
            label=server_label(method, k),
        )
        curves[label] = dict(points)
    ax.axhline(1.0, color=BASELINE_GRAY, lw=1)
    ax.annotate("no speculation", (xs[-1], 1.0), fontsize=7, va="bottom", ha="right")
    ax.set(xscale="log", xlabel="concurrent users", ylabel="tokens/s ÷ no speculation")
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.legend(fontsize=7, ncols=2)
    caption = "Speculation pays for one user and costs throughput on a busy server"
    best = max(curves, key=lambda x: curves[x].get(1, 0)) if curves else None
    if best and 64 in curves[best]:
        _, method, k = parse_label(best)
        caption = (
            f"The best method for one user ({server_label(method, k)}, {curves[best][1]:.2f}×) gives "
            f"{curves[best][64]:.2f}× at 64 users"
        )
        below = [x for x in curves if curves[x].get(64, 1.0) < 1.0]
        caption += (
            f", and {len(below)} of the {len(curves)} setups fall below no speculation there: speculation "
            "spends spare compute, and a busy server has little."
        )
    return fig, caption


def speedup_surface(m6: Records, method: str = "draft") -> tuple[plt.Figure, str]:
    """Speedup of one method over (draft length, users): where is the sweet spot?"""
    labels = [x for x in _spec_labels(m6) if parse_label(x)[1] == method]
    ks = sorted(parse_label(x)[2] for x in labels)
    users = [1, *USERS]
    grid = np.full((len(ks), len(users)), np.nan)
    for i, k in enumerate(ks):
        label = f"bf16-{method}-k{k}"
        values = [one_user_speedup(m6, label), *(speedup(m6, label, "spec_mixed", u) for u in USERS)]
        grid[i] = [np.nan if v is None else v for v in values]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    span = max(0.05, float(np.nanmax(np.abs(grid - 1))))
    image = ax.imshow(grid, cmap="PuOr_r", vmin=1 - span, vmax=1 + span, aspect="auto")
    for i in range(len(ks)):
        for j in range(len(users)):
            if not np.isnan(grid[i, j]):
                dark = abs(grid[i, j] - 1) > 0.7 * span  # the colormap's ends are dark: switch to white text
                color = "white" if dark else "black"
                ax.text(j, i, f"{grid[i, j]:.2f}×", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(users)), [str(u) for u in users])
    ax.set_yticks(range(len(ks)), [str(k) for k in ks])
    ax.set(xlabel="concurrent users", ylabel="draft length k")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="tokens/s ÷ no speculation")
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    worst = np.unravel_index(np.nanargmin(grid), grid.shape)
    caption = (
        f"With the Qwen3-0.6B drafter the best cell is k = {ks[best[0]]} for {_users(users[best[1]])} "
        f"({grid[best]:.2f}×) and the worst k = {ks[worst[0]]} for {_users(users[worst[1]])} "
        f"({grid[worst]:.2f}×): "
        "longer drafts and busier servers both waste more verified tokens."
    )
    return fig, caption


def waterfall_v3(m6: Records, dollars_per_hour: float = 0.80) -> tuple[plt.Figure, str]:
    """$ per 1M output tokens with each speculative method, for one user and for 64 users."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    regimes = [
        ("one user", lambda label: _mean([tok_s(m6, label, w) for w in ONE_USER])),
        ("64 users", lambda label: tok_s(m6, label, "spec_mixed", 64)),
    ]
    found = {}
    for ax, (title, rate) in zip(axes, regimes, strict=True):
        base = rate("bf16-none")
        if not base:
            ax.axis("off")
            continue
        cost = lambda r: dollars_per_hour / (r * 3600) * 1e6  # noqa: E731
        bars = [("BF16", cost(base))]
        for label in _spec_labels(m6):
            r = rate(label)
            if r:
                _, method, k = parse_label(label)
                bars.append((f"{SHORT[method]}\nk = {k}", cost(r)))
        for i, (_name, c) in enumerate(bars):
            if i == 0:
                ax.bar(i, c, color=BASELINE_GRAY)
                continue
            delta = c - bars[0][1]
            ax.bar(
                i,
                delta,
                bottom=bars[0][1],
                color=OKABE_ITO["green"] if delta < 0 else OKABE_ITO["vermillion"],
            )
            ax.annotate(
                f"{delta / bars[0][1]:+.0%}", (i, max(c, bars[0][1])), ha="center", va="bottom", fontsize=8
            )
        ax.set_xticks(range(len(bars)), [n for n, _ in bars], fontsize=6.5)
        ax.set(title=f"Qwen3-1.7B, {title}", ylabel="$ per 1M output tokens")
        ax.set_ylim(0, max(c for _, c in bars) * 1.15)
        changes = [c / bars[0][1] - 1 for _, c in bars[1:]]
        if changes:
            found[title] = (min(changes), max(changes))
    fig.tight_layout()
    caption = "Waterfall v3: speculative decoding against BF16 on Qwen3-1.7B"
    if "one user" in found and "64 users" in found:
        caption += (
            f": for one user the best method changes cost by {found['one user'][0]:+.0%}; at 64 users every "
            "method "
            f"lands between {found['64 users'][0]:+.0%} and {found['64 users'][1]:+.0%}"
        )
    return fig, caption + "."


def make_all(m6: Records, out_dir: str | Path) -> list[Path]:
    apply_style()
    figures = {
        "m6_highlight": lambda: highlight(m6),
        "m6_accepted_lengths": lambda: accepted_lengths(m6),
        "m6_theory": lambda: theory_vs_measured(m6),
        "m6_speedup_vs_users": lambda: speedup_vs_users(m6),
        "m6_speedup_surface": lambda: speedup_surface(m6),
        "m6_waterfall": lambda: waterfall_v3(m6),
    }
    written: list[Path] = []
    for name, build in figures.items():
        try:
            fig, caption = build()
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as err:  # data for this figure not in yet
            print(f"skipping {name}: {type(err).__name__}: {err}")
            continue
        written += save_figure(fig, name, caption, out_dir)
    if any(m["target"] == "bf16" for m in _newest(m6, "m6_highlight")):
        page = Path(out_dir) / "m6_highlight.html"
        page.write_text(highlight_html(m6), encoding="utf-8")
        written.append(page)
    return written
