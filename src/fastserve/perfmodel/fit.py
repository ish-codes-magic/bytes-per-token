"""Calibrating and checking the serving model against measured points. Stdlib only.

A `Point` is one measured load on one configuration: what was run (`stack`, `load`) and what came out.
`calibrate` fits each constant of `Calibration` on a named group of points, in stages, and nothing else:
every other point is a check. The stages, and which points each may use, are the whole method:

    prefill      share of peak FLOP/s in prefill (linear, attention)     one-user long prompts: TTFT
    overhead     fixed cost of a step                                    BF16 decode sweep, batch ≤ 16
    request      per-request handling before prefill                     the same sweep, one user: TTFT
    act_quant    W8A8's extra per step                                   FP8/INT8 decode sweep, batch ≤ 16
    flash_attn   per-sequence cost; FlashAttention's loss with batch     BF16 at ≥ 64 users: decode,
                                                                         capacity, saturation
    flashinfer   FlashInfer's efficiency on BF16 and on FP8 KV           FlashInfer servers
    draft        cost per drafted token: fixed, and per sequence         EAGLE-3, one user and 64 users

The fit minimizes squared log error of the predicted quantity, one or two constants at a time, by
golden-section search: no optimizer library, and every stage is small enough to check by hand.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from fastserve.perfmodel.serving import Calibration, Hardware, Load, Stack, predict

GOLDEN = (math.sqrt(5) - 1) / 2


@dataclass(frozen=True)
class Point:
    """One measured load. `group` names what the point is, so stages and tables can select by it."""

    source: str  # the milestone that measured it: "m4", "m5", "m6", "m8"
    group: str  # e.g. "decode", "long", "capacity", "saturation", "spec"
    model: str
    label: str  # the server's label in its own campaign
    workload: str
    users: int  # the load's nominal concurrency; `load.users` is what was in flight on average
    stack: Stack
    load: Load
    tok_s: float
    tpot_ms: float | None = None
    ttft_ms: float | None = None


def log_error(predicted: float, measured: float) -> float:
    return math.log(predicted / measured)


def golden_section(f: Callable[[float], float], lo: float, hi: float, iterations: int = 60) -> float:
    """The x in [lo, hi] minimizing a function with one minimum there."""
    a, b = lo, hi
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = f(d)
    return (a + b) / 2


def minimize(
    loss: Callable[[dict[str, float]], float], bounds: dict[str, tuple[float, float]], rounds: int = 40
):
    """Coordinate descent: golden-section on one parameter at a time, for a few rounds. Returns the best."""
    x = {name: (lo + hi) / 2 for name, (lo, hi) in bounds.items()}
    for _ in range(rounds):
        for name, (lo, hi) in bounds.items():
            x[name] = golden_section(lambda v, name=name: loss({**x, name: v}), lo, hi)
    return x


def predicted(point: Point, configs: dict[str, Any], hw: Hardware, cal: Calibration) -> dict[str, float]:
    return predict(configs[point.model], hw, point.stack, cal, point.load)


def squared_log_error(
    points: list[Point], configs: dict[str, Any], hw: Hardware, cal: Calibration, metric: str = "tok_s"
) -> float:
    total = 0.0
    for p in points:
        measured = getattr(p, metric)
        if measured:
            total += log_error(predicted(p, configs, hw, cal)[metric], measured) ** 2
    return total


def _fit(points, configs, hw, cal, bounds, build, metric="tok_s") -> Calibration:
    """Fit the constants named in `bounds`; `build(cal, values)` puts them into a Calibration."""
    if not points:
        return cal
    best = minimize(lambda v: squared_log_error(points, configs, hw, build(cal, v), metric), bounds)
    return build(cal, best)


def select(points: list[Point], **want: Any) -> list[Point]:
    """Points whose fields (or stack fields) equal the wanted values; a tuple means "any of"."""

    def ok(p: Point) -> bool:
        for name, value in want.items():
            got = getattr(p, name) if hasattr(p, name) else getattr(p.stack, name)
            if got not in (value if isinstance(value, tuple) else (value,)):
                return False
        return True

    return [p for p in points if ok(p)]


def stages(points: list[Point]) -> dict[str, list[Point]]:
    """Which points each calibration stage may use. Everything not listed here is held out."""
    plain = [p for p in points if not p.stack.prefix_caching and p.stack.speculation is None]
    fa = select(plain, backend="flash_attn", kv="bf16")
    small = [p for p in select(fa, group="decode") if p.users <= 16]
    eagle = [p for p in points if p.stack.speculation is not None and p.label.endswith("eagle3-k3")]
    return {
        "prefill": [p for p in select(fa, group="long", weights="bf16") if p.ttft_ms],
        "prefill_formats": [p for p in select(fa, group="long") if p.ttft_ms and p.stack.weights != "bf16"],
        "overhead": select(small, weights="bf16"),
        "request": [p for p in select(small, weights="bf16") if p.ttft_ms],
        "act_quant": select(small, weights=("fp8", "int8")),
        "flash_attn": [
            p for p in select(fa, weights="bf16", group=("decode", "capacity", "saturation")) if p.users >= 64
        ],
        "flashinfer_bf16": select(plain, backend="flashinfer", kv="bf16", weights="bf16"),
        "flashinfer_fp8": select(plain, backend="flashinfer", kv="fp8", weights="bf16"),
        "draft_step": [p for p in eagle if p.users == 1],
        "draft_per_seq": [p for p in eagle if p.users == 64],
    }


def calibrate(points: list[Point], configs: dict[str, Any], hw: Hardware, kv_slack: float) -> Calibration:
    """Fit every constant on its own stage's points. Two passes, since prefill and step lean on each other."""
    use = stages(points)
    formats = sorted({p.stack.weights for p in use["prefill_formats"]})
    cal = Calibration(kv_slack=kv_slack, prefill_linear={"bf16": 1.0, **dict.fromkeys(formats, 1.0)})
    for _ in range(2):
        cal = _fit(
            use["prefill"],
            configs,
            hw,
            cal,
            {"linear": (0.2, 3.0), "attention": (0.2, 3.0)},
            lambda c, v: replace(
                c, prefill_linear={**c.prefill_linear, "bf16": v["linear"]}, prefill_attention=v["attention"]
            ),
            metric="ttft_ms",
        )
        for fmt in formats:
            cal = _fit(
                select(use["prefill_formats"], weights=fmt),
                configs,
                hw,
                cal,
                {"linear": (0.2, 3.0)},
                lambda c, v, fmt=fmt: replace(c, prefill_linear={**c.prefill_linear, fmt: v["linear"]}),
                metric="ttft_ms",
            )
        cal = _fit(
            use["overhead"],
            configs,
            hw,
            cal,
            {"step_s": (0.0, 5e-3)},
            lambda c, v: replace(c, step_s=v["step_s"]),
        )
        cal = _fit(
            use["request"],
            configs,
            hw,
            cal,
            {"request_s": (0.0, 0.05)},
            lambda c, v: replace(c, request_s=v["request_s"]),
            metric="ttft_ms",
        )
        cal = _fit(
            use["act_quant"],
            configs,
            hw,
            cal,
            {"act_quant_s": (0.0, 3e-3)},
            lambda c, v: replace(c, act_quant_s=v["act_quant_s"]),
        )
        cal = _fit(
            use["flash_attn"],
            configs,
            hw,
            cal,
            {"penalty": (0.0, 0.5), "per_seq_s": (0.0, 3e-4)},
            lambda c, v: replace(c, flash_attn_batch_penalty=v["penalty"], per_seq_s=v["per_seq_s"]),
        )
        for kv in ("bf16", "fp8"):
            cal = _fit(
                use[f"flashinfer_{kv}"],
                configs,
                hw,
                cal,
                {"efficiency": (0.3, 1.2)},
                lambda c, v, kv=kv: replace(
                    c, flashinfer_efficiency={**c.flashinfer_efficiency, kv: v["efficiency"]}
                ),
            )
        cal = _fit(
            use["draft_step"],
            configs,
            hw,
            cal,
            {"draft_step_s": (0.0, 3e-3)},
            lambda c, v: replace(c, draft_step_s=v["draft_step_s"]),
        )
        cal = _fit(
            use["draft_per_seq"],
            configs,
            hw,
            cal,
            {"draft_per_seq_s": (0.0, 3e-4)},
            lambda c, v: replace(c, draft_per_seq_s=v["draft_per_seq_s"]),
        )
    return cal


def fitted_on(points: list[Point]) -> set[int]:
    """ids of the points some stage used: the rest are held out."""
    return {id(p) for group in stages(points).values() for p in group}


def errors(
    points: list[Point], configs: dict[str, Any], hw: Hardware, cal: Calibration, metric: str = "tok_s"
) -> list[float]:
    """Signed relative errors (predicted / measured − 1) of the points that have this metric."""
    out = []
    for p in points:
        measured = getattr(p, metric)
        if measured:
            out.append(predicted(p, configs, hw, cal)[metric] / measured - 1)
    return out


def median_abs(values: list[float]) -> float | None:
    return statistics.median(abs(v) for v in values) if values else None
