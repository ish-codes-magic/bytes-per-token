"""M8 analysis: the performance model's inputs and checks, and the full-stack ablation. Stdlib only.

Two halves:

- **Before M8's runs.** Every load measured in M2–M6 becomes a `Point`. The serving model
  (perfmodel/serving.py) is calibrated on named subsets of them and checked on the rest. Its predictions for
  every server of the M8 plan are then frozen in benchmarks/predictions/m8_model.json.
- **After.** M8's own records: the ladder, leave-one-out, the interaction of every pair, energy and cost, and
  the frozen predictions against what was measured.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from fastserve.perfmodel import fit
from fastserve.perfmodel.fit import Point
from fastserve.perfmodel.serving import (
    FLASH_ATTN,
    FLASHINFER,
    Calibration,
    Hardware,
    Load,
    Speculation,
    Stack,
    linear_params,
    predict,
    weight_bytes,
)
from fastserve.report.m2 import token_weighted_context
from fastserve.report.tables import markdown_table
from fastserve.serving.ablation import BASE, expand, letters

Records = list[dict[str, Any]]
DASH = "—"
SMALL, LARGE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"
M4_WEIGHTS = {"bf16": "bf16", "fp8": "fp8", "int8": "int8", "gptq": "int4", "awq": "int4"}
# The result files the model is calibrated on: everything measured before M8.
EARLIER = {"m2": "m2_serving", "m4": "m4_production", "m5": "m5_kv", "m6": "m6_spec"}
GROUPS = {
    "decode": "decode",
    "chat": "decode",
    "throughput": "throughput",
    "saturation": "saturation",
    "capacity": "capacity",
    "long_8k": "long",
    "long_16k": "long",
    "long_32k": "long",
    "multi_turn": "prefix",
    "shared_prefix": "prefix",
    "m8_latency": "spec",
}


def _f(x: float | None, digits: int = 2, suffix: str = "") -> str:
    return DASH if x is None else f"{x:,.{digits}f}{suffix}"


def _columns(table: dict[str, Any]) -> dict[str, list]:
    return {name: [row[i] for row in table["rows"]] for i, name in enumerate(table["columns"])}


def _mean(values: list[Any]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _sorted(records: Records, experiment: str) -> list[dict[str, Any]]:
    """Metrics of one experiment, oldest first, so a later record of the same thing overwrites."""
    found = [r for r in sorted(records, key=lambda r: r["timestamp"]) if r["experiment"] == experiment]
    return [r["metrics"] for r in found]


def hardware(hw_records: Records) -> Hardware:
    """M0's ceilings: read bandwidth over large transfers, the best matmul rate per format, the L2 size."""
    from fastserve.hw.analysis import gpu_spec, measured_bandwidth, measured_peak_flops

    return Hardware(
        bandwidth=measured_bandwidth(hw_records),
        peak=measured_peak_flops(hw_records),
        l2_bytes=(gpu_spec(hw_records) or {}).get("l2_bytes", 0),
    )


# ---- measured loads → points -------------------------------------------------------------------------------


def kv_tokens_by_label(records: Records) -> dict[tuple[str, str], float]:
    """(model, server label) → the KV cache's capacity in tokens, from each server's own startup log."""
    return {
        (m["model"], m["label"]): m["kv_cache_tokens"]
        for m in _sorted(records, "server_start")
        if m.get("kv_cache_tokens")
    }


def in_flight(m: dict[str, Any]) -> float:
    """Requests in the system on average during a load: Σ (finished − sent) ÷ the load's duration.

    A closed loop of N users keeps fewer than N in flight: the run's tail has only stragglers, more so when
    lengths vary. Throughput follows this number, not N.
    """
    users = m["load"]["concurrency"]
    c = _columns(m["requests"])
    pairs = zip(c.get("sent", []), c.get("finished", []), strict=False)
    spans = [(sent, done) for sent, done in pairs if sent is not None and done]
    if users <= 1 or not spans:
        return float(users)
    duration = max(done for _, done in spans) - min(sent for sent, _ in spans)
    return min(sum(done - sent for sent, done in spans) / duration, float(users))


def measured_load(
    m: dict[str, Any], kv_tokens: float | None, workload: dict[str, Any] | None = None
) -> Load | None:
    """The averages of what a serving record actually sent and received (closed-loop loads only).

    `workload` is its definition, for what the requests cannot show: how many sequences share a prefix.
    """
    if m["load"].get("mode") != "closed":
        return None
    c = _columns(m["requests"])
    done = [i for i, tokens in enumerate(c["output_tokens"]) if tokens]
    if not done:
        return None
    prompt = sum(c["prompt_len"][i] for i in done) / len(done)
    output = sum(c["output_tokens"][i] for i in done) / len(done)
    cached = _mean([c["cached_tokens"][i] for i in done]) if "cached_tokens" in c else None
    shared = (workload or {}).get("shared_prefix_len", 0)
    return Load(
        users=in_flight(m),
        prompt_len=prompt,
        output_len=output,
        cached_len=cached or 0.0,
        kv_tokens=kv_tokens,
        context=token_weighted_context(m["requests"]),
        shared_len=shared,
        sharers=m["load"]["concurrency"] / (workload or {}).get("prefixes", 1) if shared else 1.0,
    )


def tokens_per_pass(m: dict[str, Any]) -> float | None:
    """Tokens kept per target pass during this load, from vLLM's counters.

    Every pass yields the target's own token plus the draft tokens it accepted, so with G tokens generated
    and A accepted there were G − A passes: G / (G − A) tokens each. Counting passes this way includes the
    ones that had nothing to verify (an n-gram drafter with no match), which the drafts counter leaves out.
    """
    timeline = m.get("server_timeline") or {}
    c = _columns(timeline) if timeline.get("rows") else {}
    if "spec_accepted" not in c:
        return None
    accepted = [v for v in c["spec_accepted"] if v is not None]
    generated = [v for v in c["generation_tokens"] if v is not None]
    if len(accepted) < 2 or generated[-1] == generated[0]:
        return None
    tokens, kept = generated[-1] - generated[0], accepted[-1] - accepted[0]
    return tokens / (tokens - kept) if tokens > kept else None


def eagle_bytes(cfg: Any, head: dict[str, Any]) -> float:
    """What an EAGLE-3 head reads per drafted token: its decoder layer(s) and its reduced-vocabulary head."""
    layer = linear_params(cfg) / cfg.num_layers
    return 2 * (head["layers"] * layer + head["draft_vocab_size"] * cfg.hidden_size)


def _point(source: str, m: dict[str, Any], stack: Stack, load: Load | None) -> Point | None:
    summary = m["summary"]
    if load is None or not summary.get("output_throughput"):
        return None
    return Point(
        source=source,
        group=GROUPS.get(m["workload"], "spec" if m["workload"].startswith("spec_") else m["workload"]),
        model=m["model"],
        label=m["server"],
        workload=m["workload"],
        users=m["load"]["concurrency"],
        stack=stack,
        load=load,
        tok_s=summary["output_throughput"],
        tpot_ms=(summary.get("tpot_ms") or {}).get("p50"),
        ttft_ms=(summary.get("ttft_ms") or {}).get("p50") if m["load"]["concurrency"] == 1 else None,
    )


def _points(source: str, records: Records, stack_for, workloads: dict[str, Any] | None = None) -> list[Point]:
    capacity = kv_tokens_by_label(records)
    latest: dict[tuple, Point] = {}
    for m in _sorted(records, "serving"):
        stack = stack_for(m)
        if stack is None:
            continue
        kv_tokens = capacity.get((m["model"], m["server"]))
        point = _point(source, m, stack, measured_load(m, kv_tokens, (workloads or {}).get(m["workload"])))
        if point is not None:
            latest[(m["model"], m["server"], m["workload"], point.users)] = point
    return list(latest.values())


def m2_points(m2: Records) -> list[Point]:
    return _points("m2", m2, lambda m: Stack())


def m4_points(m4: Records) -> list[Point]:
    return _points("m4", m4, lambda m: Stack(weights=M4_WEIGHTS[m["server"]]))


def m5_points(m5: Records, workloads: dict[str, Any]) -> list[Point]:
    stacks = {
        "bf16kv": Stack(),
        "bf16kv-flashinfer": Stack(backend=FLASHINFER),
        "fp8kv": Stack(kv="fp8", backend=FLASHINFER),
        "fp8w-fp8kv": Stack(weights="fp8", kv="fp8", backend=FLASHINFER),
        "prefix-off": Stack(),
        "prefix-on": Stack(prefix_caching=True),
    }
    return _points("m5", m5, lambda m: stacks.get(m["server"]), workloads)


def m6_points(m6: Records, configs: dict[str, Any], head: dict[str, Any]) -> list[Point]:
    """M6's servers `<target>-<method>[-k<n>]`. A drafter's cost per token is the bytes it reads."""
    targets = {"bf16": "bf16", "fp8": "fp8", "awq": "int4"}

    def stack_for(m: dict[str, Any]) -> Stack | None:
        target, method, *rest = m["server"].split("-")
        stack = Stack(weights=targets[target])
        if method == "none":
            return stack
        kept = tokens_per_pass(m)
        if kept is None:
            return None
        draft = {
            "eagle3": eagle_bytes(configs[m["model"]], head),
            "draft": weight_bytes(configs[SMALL], "bf16"),
            "draftawq": weight_bytes(configs[SMALL], "int4"),
            "ngram": 0.0,
        }[method]
        return replace(stack, speculation=Speculation(int(rest[0][1:]), kept, draft))

    points = _points("m6", m6, stack_for)
    # A separate draft model inside vLLM costs more per drafted token than the bytes it reads (M6's open
    # question, worst with a quantized target). The model does not capture that: these points are kept apart.
    return [replace(p, group="draft_model") if "-draft" in p.label else p for p in points]


def kv_slack(points: list[Point], m5: Records) -> float:
    """Cache tokens a running sequence holds ÷ (prompt + output), from the capacity runs' own counters."""
    running = {}
    for m in _sorted(m5, "serving"):
        if m["workload"] == "capacity":
            running[(m["model"], m["server"])] = _mean(_columns(m["server_timeline"])["running"][2:-2])
    ratios = []
    for p in fit.select(points, source="m5", group="capacity"):
        seqs = running.get((p.model, p.label))
        if seqs and p.load.kv_tokens:
            ratios.append(p.load.kv_tokens / (seqs * (p.load.prompt_len + p.load.output_len)))
    return sum(ratios) / len(ratios)


def earlier_points(
    m2: Records,
    m4: Records,
    m5: Records,
    m6: Records,
    configs: dict[str, Any],
    head: dict[str, Any],
    workloads: dict[str, Any],
) -> list[Point]:
    return [*m2_points(m2), *m4_points(m4), *m5_points(m5, workloads), *m6_points(m6, configs, head)]


# ---- the model, calibrated and checked ---------------------------------------------------------------------


def calibrated(points: list[Point], m5: Records, configs: dict[str, Any], hw: Hardware) -> Calibration:
    return fit.calibrate(points, configs, hw, kv_slack(points, m5))


def calibration_table(cal: Calibration, points: list[Point]) -> str:
    """Every fitted constant, its value, and the measured points it was fitted on."""
    use = fit.stages(points)
    n = {name: len(group) for name, group in use.items()}
    ms, us = 1e3, 1e6
    big = f"{n['flash_attn']} (BF16, ≥ 64 users: decode, capacity, saturation)"
    rows = [
        ["Fixed cost of a decode step", _f(cal.step_s * ms, 2, " ms"), f"{n['overhead']} (BF16, ≤ 16 users)"],
        ["Added per sequence in the batch", _f(cal.per_seq_s * us, 0, " µs"), big],
        [
            "FlashAttention: efficiency lost per doubling of the batch",
            _f(cal.flash_attn_batch_penalty, 3),
            big,
        ],
        [
            "W8A8's extra per step",
            _f(cal.act_quant_s * ms, 2, " ms"),
            f"{n['act_quant']} (FP8/INT8, ≤ 16 users)",
        ],
        [
            "FlashInfer's share of the bandwidth, BF16 cache",
            _f(100 * cal.flashinfer_efficiency["bf16"], 0, "%"),
            f"{n['flashinfer_bf16']} (M5's FlashInfer control)",
        ],
        [
            "FlashInfer's share of the bandwidth, FP8 cache",
            _f(100 * cal.flashinfer_efficiency["fp8"], 0, "%"),
            f"{n['flashinfer_fp8']} (M5's FP8 KV servers)",
        ],
        [
            "Prefill: share of BF16's peak FLOP/s, linear layers",
            _f(100 * cal.prefill_linear["bf16"], 0, "%"),
            f"{n['prefill']} (one user, 8k–32k prompts: TTFT)",
        ],
        [
            "Prefill: share of BF16's peak FLOP/s, attention",
            _f(100 * cal.prefill_attention, 0, "%"),
            "the same",
        ],
    ]
    for fmt, share in sorted(cal.prefill_linear.items()):
        if fmt != "bf16":
            count = len(fit.select(use["prefill_formats"], weights=fmt))
            name = f"Prefill: share of its own peak FLOP/s, {fmt.upper()} layers"
            rows.append([name, _f(100 * share, 0, "%"), f"{count} (one user, long prompts: TTFT)"])
    rows += [
        [
            "Per request, before its prefill",
            _f(cal.request_s * ms, 1, " ms"),
            f"{n['request']} (one user: TTFT)",
        ],
        [
            "Fixed cost per drafted token",
            _f(cal.draft_step_s * ms, 2, " ms"),
            f"{n['draft_step']} (EAGLE-3, one user)",
        ],
        [
            "Per drafted token and sequence",
            _f(cal.draft_per_seq_s * us, 0, " µs"),
            f"{n['draft_per_seq']} (EAGLE-3, 64 users)",
        ],
        [
            "Cache tokens a running sequence holds ÷ its length",
            _f(cal.kv_slack, 3),
            "the capacity runs' counters",
        ],
    ]
    return markdown_table(["Constant", "Value", "Fitted on (points)"], rows)


WORKLOAD_LABELS = {
    "m8_latency": "Latency (1 user, real prompts)",
    "spec_mixed": "Busy (64 users, real prompts)",
    "capacity": "Capacity (96 users, 4k-token prompts)",
    "multi_turn": "Multi-turn (8 users, shared prefixes)",
    "long_32k": "Long (1 user, 32k tokens)",
}
SOURCE_LABELS = {
    "m2": "M2 baselines",
    "m4": "M4 weight formats",
    "m5": "M5 KV cache and prefix caching",
    "m6": "M6 speculative decoding",
    "m8": "M8 full stack",
}


LONG_PROMPT = 2000  # tokens: above this TTFT is the prefill; below it, mostly fixed costs


def _ttft_errors(points: list[Point], configs: dict[str, Any], hw: Hardware, cal: Calibration):
    """(relative errors on long prompts, absolute errors in ms on short ones) of one-user TTFT."""
    long, short = [], []
    for p in points:
        if p.ttft_ms:
            got = fit.predicted(p, configs, hw, cal)["ttft_ms"]
            if p.load.prompt_len >= LONG_PROMPT:
                long.append(got / p.ttft_ms - 1)
            else:
                short.append(got - p.ttft_ms)
    return long, short


def modeled(points: list[Point]) -> list[Point]:
    """Points the model claims to cover: everything except M6's separate draft models."""
    return [p for p in points if p.group != "draft_model"]


def validation_table(points: list[Point], configs: dict[str, Any], hw: Hardware, cal: Calibration) -> str:
    """Median and worst error per campaign, split into points a stage was fitted on and points held out."""
    used = fit.fitted_on(points)
    groups = []
    for source, name in SOURCE_LABELS.items():
        mine = [p for p in modeled(points) if p.source == source]
        groups.append((name, "used in a fit", [p for p in mine if id(p) in used]))
        groups.append((name, "held out", [p for p in mine if id(p) not in used]))
    everything = modeled(points)
    groups.append(("**All of the above**", "", everything))
    groups.append(("**All held out**", "", [p for p in everything if id(p) not in used]))
    draft = [p for p in points if p.group == "draft_model"]
    groups.append(("M6: a separate draft model (not modeled)", "held out", draft))
    rows = []
    for name, kind, subset in groups:
        errs = fit.errors(subset, configs, hw, cal)
        if not errs:
            continue
        long, short = _ttft_errors(subset, configs, hw, cal)
        rows.append(
            [
                name,
                kind,
                str(len(errs)),
                _f(100 * fit.median_abs(errs), 1, "%"),
                _f(100 * max(abs(e) for e in errs), 0, "%"),
                _f(100 * sum(abs(e) <= 0.15 for e in errs) / len(errs), 0, "%"),
                _f(100 * fit.median_abs(long), 1, "%") if long else DASH,
                _f(fit.median_abs(short), 0, " ms") if short else DASH,
            ]
        )
    headers = [
        "Measured in",
        "Points",
        "Count",
        "Median error, tokens/s",
        "Worst",
        "Within 15%",
        f"Median error, TTFT (prompts ≥ {LONG_PROMPT:,} tokens)",
        "Median error, TTFT (shorter prompts)",
    ]
    return markdown_table(headers, rows)


def worst_points(
    points: list[Point], configs: dict[str, Any], hw: Hardware, cal: Calibration, n: int = 8
) -> str:
    """The model's largest misses: where to look for what it leaves out."""
    scored = []
    for p in points:
        got = fit.predicted(p, configs, hw, cal)["tok_s"]
        scored.append((abs(got / p.tok_s - 1), p, got))
    rows = [
        [
            SOURCE_LABELS[p.source],
            p.model.split("/")[-1],
            p.label,
            p.workload,
            str(p.users),
            _f(p.tok_s, 0),
            _f(got, 0),
            _f(100 * (got / p.tok_s - 1), 0, "%"),
        ]
        for _, p, got in sorted(scored, key=lambda item: -item[0])[:n]
    ]
    headers = ["Measured in", "Model", "Server", "Workload", "Users", "Measured tok/s", "Predicted", "Error"]
    return markdown_table(headers, rows)


# ---- the M8 plan, predicted --------------------------------------------------------------------------------


def stack_of(label: str, cfg: Any, kept: float | None, head: dict[str, Any]) -> Stack:
    """The serving model's view of an M8 label: which letters are on."""
    on = letters(label)
    fp8_kv = "k" in on
    spec = Speculation(3, kept, eagle_bytes(cfg, head)) if "s" in on and kept else None
    return Stack(
        weights="fp8" if "w" in on else "int4" if "a" in on else "bf16",
        kv="fp8" if fp8_kv else "bf16",
        backend=FLASHINFER if fp8_kv or "f" in on else FLASH_ATTN,
        prefix_caching="p" in on,
        speculation=spec,
    )


def expected_loads(m5: Records, m6: Records, workloads: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """What each M8 workload will send, taken from the earlier runs of the same workloads.

    `kept` is the tokens per pass an EAGLE-3 head got on that kind of prompt in M6. The random-token workloads
    (capacity, multi_turn, long_32k) have no such measurement: they borrow the mixed real-prompt value.
    """

    def load_of(records: Records, server: str, names: tuple[str, ...], users: int) -> Load:
        rows: list[list] = []
        columns: list[str] = []
        running = []
        for m in _sorted(records, "serving"):
            same = m["model"] == LARGE and m["server"] == server and m["workload"] in names
            if same and m["load"].get("concurrency") == users:
                columns, rows = m["requests"]["columns"], rows + m["requests"]["rows"]
                running.append(in_flight(m))
        pooled = {
            "load": {"mode": "closed", "concurrency": users},
            "requests": {"columns": columns, "rows": rows},
        }
        load = measured_load(pooled, None, workloads.get(names[0]) if len(names) == 1 else None)
        return replace(load, users=sum(running) / len(running))

    def kept(workloads: tuple[str, ...], users: int) -> float:
        tokens = accepted = 0.0
        for m in _sorted(m6, "serving"):
            same = m["model"] == LARGE and m["server"] == "bf16-eagle3-k3" and m["workload"] in workloads
            if same and m["load"].get("concurrency") == users:
                c = _columns(m["server_timeline"])
                tokens += c["generation_tokens"][-1] - c["generation_tokens"][0]
                accepted += c["spec_accepted"][-1] - c["spec_accepted"][0]
        return tokens / (tokens - accepted)

    one_user = ("spec_chat", "spec_code", "spec_math", "spec_summarize")
    busy = kept(("spec_mixed",), 64)
    return {
        "m8_latency": {"load": load_of(m6, "bf16-none", one_user, 1), "kept": kept(one_user, 1)},
        "spec_mixed": {"load": load_of(m6, "bf16-none", ("spec_mixed",), 64), "kept": busy},
        "capacity": {
            "load": load_of(m5, "bf16kv", ("capacity",), 96),
            "kept": busy,
            "limited_by_cache": True,
        },
        "multi_turn": {
            "load": replace(
                load_of(m5, "prefix-off", ("multi_turn",), 8),
                cached_len=load_of(m5, "prefix-on", ("multi_turn",), 8).cached_len,
            ),
            "kept": busy,
        },
        "long_32k": {"load": load_of(m5, "bf16kv", ("long_32k",), 1), "kept": busy},
    }


def kv_capacity(model: str, stack: Stack, cfg: Any, starts: dict[str, Any]) -> float:
    """The KV cache's capacity in tokens for a configuration, estimated from earlier servers' startup logs.

    vLLM gives the cache what is left of its memory budget: budget − weights − what FlashInfer and a drafter
    reserve. The three terms come from the servers that differed in exactly that.
    """
    s = starts[model]
    gib = s["budget_gib"] - s["weights_gib"][stack.weights]
    gib -= s["flashinfer_gib"] if stack.backend == FLASHINFER else 0.0
    gib -= s["drafter_gib"] if stack.speculation is not None else 0.0
    bytes_per_token = cfg.kv_bytes_per_token(1 if stack.kv == "fp8" else 2)
    return gib * 2**30 / bytes_per_token


def startup_memory(
    m4: Records, m5: Records, m6: Records, configs: dict[str, Any], head: dict[str, Any]
) -> dict[str, Any]:
    """Per model: the memory budget, each weight format's size, and what FlashInfer and an EAGLE head take."""

    def start(records: Records, model: str, label: str) -> dict[str, Any]:
        return next(
            m
            for m in reversed(_sorted(records, "server_start"))
            if (m["model"], m["label"]) == (model, label)
        )

    out = {}
    for model in (SMALL, LARGE):
        bf16 = start(m4, model, "bf16")
        budget = bf16["kv_cache_memory_gib"] + bf16["model_memory_gib"]
        weights = {
            fmt: start(m4, model, label)["model_memory_gib"]
            for fmt, label in (("bf16", "bf16"), ("fp8", "fp8"), ("int4", "awq"))
        }
        flashinfer = (
            bf16["kv_cache_memory_gib"] - start(m5, model, "bf16kv-flashinfer")["kv_cache_memory_gib"]
        )
        out[model] = {"budget_gib": budget, "weights_gib": weights, "flashinfer_gib": flashinfer}
    large = out[LARGE]
    eagle = start(m6, LARGE, "bf16-eagle3-k3")
    large["drafter_gib"] = large["budget_gib"] - large["weights_gib"]["bf16"] - eagle["kv_cache_memory_gib"]
    # No EAGLE head was ever started for the small model: scale the large one's by the heads' sizes.
    ratio = eagle_bytes(configs[SMALL], head) / eagle_bytes(configs[LARGE], head)
    out[SMALL]["drafter_gib"] = large["drafter_gib"] * ratio
    return out


def plan_predictions(
    config: dict[str, Any],
    configs: dict[str, Any],
    hw: Hardware,
    cal: Calibration,
    loads: dict[str, dict[str, Any]],
    starts: dict[str, Any],
) -> list[dict[str, Any]]:
    """The model's prediction for every (server, workload) of the M8 plan."""
    head = config["techniques"]["s"]["head"]
    rows = []
    for server in expand(config):
        model, label = server["model"], server["label"]
        if label.endswith("-r2"):
            continue
        cfg = configs[model]
        for workload in server["workloads"]:
            expected = loads[workload]
            stack = stack_of(label, cfg, expected["kept"], head)
            load = expected["load"]
            if expected.get("limited_by_cache"):
                load = replace(load, kv_tokens=kv_capacity(model, stack, cfg, starts))
            out = predict(cfg, hw, stack, cal, load)
            rows.append({"model": model, "label": label, "workload": workload, **out})
    return rows


def frozen(
    config: dict[str, Any],
    configs: dict[str, Any],
    hw: Hardware,
    cal: Calibration,
    loads: dict[str, dict[str, Any]],
    starts: dict[str, Any],
) -> dict[str, Any]:
    """Everything needed to reproduce the M8 predictions, and the predictions: the JSON file's content."""
    return {
        "hardware": asdict(hw),
        "calibration": asdict(cal),
        "expected_loads": {
            name: {"load": asdict(e["load"]), "tokens_per_pass": e["kept"]} for name, e in loads.items()
        },
        "startup_memory": starts,
        "predictions": plan_predictions(config, configs, hw, cal, loads, starts),
    }


def calibration_of(frozen_file: dict[str, Any]) -> Calibration:
    """The constants as they were frozen before M8: every M8 comparison uses these, not a refit."""
    return Calibration(**frozen_file["calibration"])


def predicted_speedups(frozen_file: dict[str, Any], model: str, labels: list[str]) -> str:
    """The frozen predictions as speedups over the base server, per workload."""
    rows_by = {(r["label"], r["workload"]): r for r in frozen_file["predictions"] if r["model"] == model}
    workloads = list(dict.fromkeys(w for _, w in rows_by))
    rows = []
    for workload in workloads:
        base = rows_by[(BASE, workload)]
        cells = [WORKLOAD_LABELS.get(workload, workload), _f(base["tok_s"], 0)]
        for label in labels:
            r = rows_by.get((label, workload))
            cells.append(_f(r["tok_s"] / base["tok_s"], 2, "×") if r else DASH)
        rows.append(cells)
    return markdown_table(["Workload", "Base (tokens/s)"] + [f"`{label}`" for label in labels], rows)
