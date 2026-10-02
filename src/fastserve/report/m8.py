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


# ---- M8's own records: the ablation ------------------------------------------------------------------------

LADDER = ["base", "w", "wk", "wkp", "wkps"]  # each step adds one technique
FULL = "wkps"
TECHNIQUES = {"w": "FP8 weights", "k": "FP8 KV cache", "p": "prefix caching", "s": "speculative decoding"}
WORKLOADS = ["m8_latency", "spec_mixed", "capacity", "multi_turn", "long_32k"]


def serving(m8: Records, model: str, label: str, workload: str) -> dict[str, Any] | None:
    """The newest record of one server on one workload."""
    found = None
    for m in _sorted(m8, "serving"):
        if (m["model"], m["server"], m["workload"]) == (model, label, workload):
            found = m
    return found


def labels_run(m8: Records, model: str) -> list[str]:
    return list(dict.fromkeys(m["server"] for m in _sorted(m8, "serving") if m["model"] == model))


def tok_s(m8: Records, model: str, label: str, workload: str) -> float | None:
    m = serving(m8, model, label, workload)
    return m["summary"].get("output_throughput") if m else None


def stat(m8: Records, model: str, label: str, workload: str, metric: str) -> float | None:
    """p50 of tpot_ms or ttft_ms."""
    m = serving(m8, model, label, workload)
    return (m["summary"].get(metric) or {}).get("p50") if m else None


def speedup(m8: Records, model: str, label: str, workload: str, over: str = BASE) -> float | None:
    a, b = tok_s(m8, model, label, workload), tok_s(m8, model, over, workload)
    return a / b if a and b else None


def power(m8: Records, model: str, label: str, workload: str) -> float | None:
    m = serving(m8, model, label, workload)
    return ((m or {}).get("server_timeline", {}).get("power") or {}).get("mean_w")


def tokens_per_joule(m8: Records, model: str, label: str, workload: str) -> float | None:
    rate, watts = tok_s(m8, model, label, workload), power(m8, model, label, workload)
    return rate / watts if rate and watts else None


def dollars(rate: float | None, dollars_per_hour: float) -> float | None:
    return dollars_per_hour / (rate * 3600) * 1e6 if rate else None


def without(label: str, letter: str) -> str:
    """The label with one technique switched off: wkps without k → wps; w without w → base."""
    return label.replace(letter, "") or BASE


def ladder_table(m8: Records, model: str, dollars_per_hour: float, labels: list[str] | None = None) -> str:
    """Per workload: the base server's throughput and cost, each ladder step as a multiple of the base."""
    labels = labels or LADDER
    rows = []
    for workload in WORKLOADS:
        base = tok_s(m8, model, BASE, workload)
        if base is None:
            continue
        full = tok_s(m8, model, labels[-1], workload)
        rows.append(
            [WORKLOAD_LABELS[workload], _f(base, 0), _f(dollars(base, dollars_per_hour), 2)]
            + [_f(speedup(m8, model, label, workload), 2, "×") for label in labels[1:]]
            + [_f(dollars(full, dollars_per_hour), 2)]
        )
    headers = (
        ["Workload", "Base (tokens/s)", "Base ($ per 1M)"]
        + [f"`{label}`" for label in labels[1:]]
        + [f"`{labels[-1]}` ($ per 1M)"]
    )
    return markdown_table(headers, rows)


def step_table(m8: Records, model: str, workload: str, dollars_per_hour: float) -> str:
    """One workload down the ladder: what each added technique did to speed, latency, cost and energy."""
    steps = [BASE, "w", "wf", "wk", "wkp", "wkps", "akps"]
    names = {
        BASE: "stock BF16",
        "w": "+ FP8 weights",
        "wf": "(control: + FlashInfer, BF16 cache)",
        "wk": "+ FP8 KV cache",
        "wkp": "+ prefix caching",
        "wkps": "+ speculative decoding",
        "akps": "(branch: INT4 weights instead of FP8)",
    }
    rows = []
    previous = None
    for label in steps:
        rate = tok_s(m8, model, label, workload)
        if rate is None:
            continue
        start = next(
            (m for m in _sorted(m8, "server_start") if (m["model"], m["label"]) == (model, label)), {}
        )
        rows.append(
            [
                names[label],
                f"`{label}`",
                _f(rate, 0),
                _f(speedup(m8, model, label, workload), 2, "×"),
                _f(rate / previous, 2, "×") if previous and label in LADDER else DASH,
                _f(stat(m8, model, label, workload, "tpot_ms"), 1),
                _f(stat(m8, model, label, workload, "ttft_ms"), 0),
                _f(dollars(rate, dollars_per_hour), 2),
                _f(tokens_per_joule(m8, model, label, workload), 1),
                f"{start['kv_cache_tokens']:,}" if start.get("kv_cache_tokens") else DASH,
            ]
        )
        if label in LADDER:
            previous = rate
    headers = [
        "Step",
        "Server",
        "Tokens/s",
        "vs base",
        "vs previous step",
        "TPOT p50 (ms)",
        "TTFT p50 (ms)",
        "$ per 1M tokens",
        "Tokens per joule",
        "KV cache (tokens)",
    ]
    return markdown_table(headers, rows)


def alone_and_in_stack(
    m8: Records, model: str, letter: str, workload: str
) -> tuple[float | None, float | None]:
    """A technique's two honest numbers: added to nothing, and what the full stack loses without it."""
    return speedup(m8, model, letter, workload), speedup(
        m8, model, FULL, workload, over=without(FULL, letter)
    )


def leave_one_out_table(m8: Records, model: str) -> str:
    """Per workload and technique: its speedup alone, and the full stack's over the stack without it."""
    rows = []
    for workload in WORKLOADS:
        cells = [WORKLOAD_LABELS[workload]]
        for letter in TECHNIQUES:
            alone, in_stack = alone_and_in_stack(m8, model, letter, workload)
            cells.append(f"{_f(alone, 2, '×')} · {_f(in_stack, 2, '×')}")
        if any(c != f"{DASH} · {DASH}" for c in cells[1:]):
            rows.append(cells)
    headers = ["Workload"] + [f"{name}: alone · in the full stack" for name in TECHNIQUES.values()]
    return markdown_table(headers, rows)


def interaction(m8: Records, model: str, a: str, b: str, workload: str) -> float | None:
    """S(a + b) / (S(a) · S(b)): 1 = the gains multiply, below 1 = they compete."""
    both = speedup(m8, model, label_pair(a, b), workload)
    sa, sb = speedup(m8, model, a, workload), speedup(m8, model, b, workload)
    return both / (sa * sb) if both and sa and sb else None


def label_pair(a: str, b: str) -> str:
    return "".join(letter for letter in TECHNIQUES if letter in (a, b))


def pairs() -> list[tuple[str, str]]:
    letters_ = list(TECHNIQUES)
    return [(a, b) for i, a in enumerate(letters_) for b in letters_[i + 1 :]]


def interaction_table(m8: Records, model: str) -> str:
    rows = []
    for a, b in pairs():
        cells = [f"{TECHNIQUES[a]} + {TECHNIQUES[b]}", f"`{label_pair(a, b)}`"]
        cells += [_f(interaction(m8, model, a, b, w), 2) for w in WORKLOADS[:4]]
        rows.append(cells)
    return markdown_table(["Pair", "Server"] + [WORKLOAD_LABELS[w] for w in WORKLOADS[:4]], rows)


def repeat_differences(m8: Records, model: str) -> list[tuple[str, str, float]]:
    """(label, workload, second run ÷ first − 1) for every server that ran twice."""
    out = []
    for label in labels_run(m8, model):
        if label.endswith("-r2"):
            for workload in WORKLOADS:
                first, second = tok_s(m8, model, label[:-3], workload), tok_s(m8, model, label, workload)
                if first and second:
                    out.append((label[:-3], workload, second / first - 1))
    return out


def repeat_table(m8: Records, model: str) -> str:
    rows = [
        [f"`{label}`", WORKLOAD_LABELS[workload], _f(100 * diff, 1, "%")]
        for label, workload, diff in repeat_differences(m8, model)
    ]
    return markdown_table(["Server", "Workload", "Second run vs first, tokens/s"], rows)


def kept_table(m8: Records, model: str) -> str:
    """Tokens per target pass with the EAGLE head, per workload and stack: what speculation worked with."""
    labels = [
        label for label in ("s", "ws", "ks", "ps", "wkps", "akps") if serving(m8, model, label, "m8_latency")
    ]
    rows = []
    for workload in WORKLOADS:
        cells = [WORKLOAD_LABELS[workload]]
        for label in labels:
            m = serving(m8, model, label, workload)
            cells.append(_f(tokens_per_pass(m) if m else None, 2))
        rows.append(cells)
    return markdown_table(["Workload"] + [f"`{label}`" for label in labels], rows)


def energy_table(m8: Records, model: str) -> str:
    rows = []
    for workload in WORKLOADS:
        base, full = (tokens_per_joule(m8, model, label, workload) for label in (BASE, FULL))
        if base is None:
            continue
        rows.append(
            [
                WORKLOAD_LABELS[workload],
                _f(power(m8, model, BASE, workload), 0),
                _f(power(m8, model, FULL, workload), 0),
                _f(base, 2),
                _f(full, 2),
                _f(full / base if full else None, 2, "×"),
            ]
        )
    headers = [
        "Workload",
        "Power, base (W)",
        "Power, full stack (W)",
        "Tokens per joule, base",
        "Full stack",
        "Change",
    ]
    return markdown_table(headers, rows)


def failures_table(m8: Records) -> str:
    rows = [[f"`{m['name']}`", m["error"][:160]] for m in _sorted(m8, "m8_failure")]
    return markdown_table(["Server", "Error"], rows) if rows else "*Every server of the plan ran.*"


# ---- the frozen predictions against M8 ---------------------------------------------------------------------


def prediction_errors(m8: Records, frozen_file: dict[str, Any]) -> list[dict[str, Any]]:
    """Every frozen prediction that has a measurement: predicted and measured tokens/s, and the error."""
    rows = []
    for pred in frozen_file["predictions"]:
        got = tok_s(m8, pred["model"], pred["label"], pred["workload"])
        if got:
            rows.append({**pred, "measured": got, "error": pred["tok_s"] / got - 1})
    return rows


def _median_abs(values: list[float]) -> float | None:
    return fit.median_abs(values)


def model_error_table(m8: Records, frozen_file: dict[str, Any]) -> str:
    """The frozen predictions' error on M8, by workload and by whether speculation was on."""
    rows_ = prediction_errors(m8, frozen_file)
    groups: list[tuple[str, list[dict[str, Any]]]] = [
        (WORKLOAD_LABELS[w], [r for r in rows_ if r["workload"] == w]) for w in WORKLOADS
    ]
    groups += [
        ("Servers without speculation", [r for r in rows_ if "s" not in letters(r["label"])]),
        ("Servers with speculation", [r for r in rows_ if "s" in letters(r["label"])]),
        ("Qwen3-1.7B", [r for r in rows_ if r["model"] == LARGE]),
        ("Qwen3-0.6B", [r for r in rows_ if r["model"] == SMALL]),
        ("**All**", rows_),
    ]
    rows = []
    for name, group in groups:
        errs = [r["error"] for r in group]
        if not errs:
            continue
        rows.append(
            [
                name,
                str(len(errs)),
                _f(100 * _median_abs(errs), 1, "%"),
                _f(100 * sorted(errs)[len(errs) // 2], 1, "%"),
                _f(100 * max(abs(e) for e in errs), 0, "%"),
                _f(100 * sum(abs(e) <= 0.15 for e in errs) / len(errs), 0, "%"),
            ]
        )
    headers = ["Points", "Count", "Median error", "Median signed error", "Worst", "Within 15%"]
    return markdown_table(headers, rows)


def worst_predictions(m8: Records, frozen_file: dict[str, Any], n: int = 10) -> str:
    rows_ = sorted(prediction_errors(m8, frozen_file), key=lambda r: -abs(r["error"]))[:n]
    rows = [
        [
            r["model"].split("/")[-1],
            f"`{r['label']}`",
            WORKLOAD_LABELS[r["workload"]],
            _f(r["measured"], 0),
            _f(r["tok_s"], 0),
            _f(100 * r["error"], 0, "%"),
        ]
        for r in rows_
    ]
    return markdown_table(["Model", "Server", "Workload", "Measured tok/s", "Predicted", "Error"], rows)


def m8_points(
    m8: Records, config: dict[str, Any], configs: dict[str, Any], workloads: dict[str, Any]
) -> list[Point]:
    """M8's loads as points with their *measured* inputs (lengths, in-flight requests, tokens per pass, cache
    size): what the model says when it is told what actually happened, instead of what was expected."""
    head = config["techniques"]["s"]["head"]

    def stack_for(m: dict[str, Any]) -> Stack | None:
        on = letters(m["server"])
        kept = tokens_per_pass(m) if "s" in on else None
        if "s" in on and kept is None:
            return None
        return stack_of(m["server"], configs[m["model"]], kept, head)

    points = _points("m8", m8, stack_for, workloads)
    # Only the capacity workload is limited by the cache; elsewhere its size is irrelevant.
    return [
        p if p.workload == "capacity" else replace(p, load=replace(p.load, kv_tokens=None)) for p in points
    ]


def informed_error_table(points: list[Point], configs: dict[str, Any], hw: Hardware, cal: Calibration) -> str:
    """The same frozen constants, fed M8's measured inputs: separates wrong physics from wrong assumptions."""
    rows = []
    for workload in WORKLOADS:
        subset = [p for p in points if p.workload == workload and not p.label.endswith("-r2")]
        errs = fit.errors(subset, configs, hw, cal)
        if errs:
            rows.append(
                [
                    WORKLOAD_LABELS[workload],
                    str(len(errs)),
                    _f(100 * _median_abs(errs), 1, "%"),
                    _f(100 * max(abs(e) for e in errs), 0, "%"),
                    _f(100 * sum(abs(e) <= 0.15 for e in errs) / len(errs), 0, "%"),
                ]
            )
    everything = fit.errors([p for p in points if not p.label.endswith("-r2")], configs, hw, cal)
    if everything:
        rows.append(
            [
                "**All**",
                str(len(everything)),
                _f(100 * _median_abs(everything), 1, "%"),
                _f(100 * max(abs(e) for e in everything), 0, "%"),
                _f(100 * sum(abs(e) <= 0.15 for e in everything) / len(everything), 0, "%"),
            ]
        )
    return markdown_table(["Workload", "Count", "Median error", "Worst", "Within 15%"], rows)


# ---- quality and cost --------------------------------------------------------------------------------------


def quality_table(m8: Records, m5: Records, m4: Records, m2_quality: Records) -> str:
    """The lossy parts of the stack, alone and together, in vLLM: perplexity, needle recall, task scores."""
    from fastserve.report import m4 as m4_report

    def one(records: Records, experiment: str, **match: Any) -> dict[str, Any]:
        found = [m for m in _sorted(records, experiment) if all(m.get(k) == v for k, v in match.items())]
        return found[-1] if found else {}

    rows = []
    for model in (SMALL, LARGE):
        stacks = [
            (
                "BF16",
                one(m4, "m4_vllm_perplexity", model=model, format="bf16"),
                one(m2_quality, "quality_needle", model=model),
                m4_report.tasks(m4, m2_quality, model, "bf16"),
            ),
            (
                "FP8 weights (`w`)",
                one(m4, "m4_vllm_perplexity", model=model, format="fp8"),
                {},
                m4_report.tasks(m4, m2_quality, model, "fp8"),
            ),
            (
                "FP8 KV cache (`k`)",
                one(m5, "m5_vllm_perplexity", model=model),
                one(m5, "m5_vllm_needle", model=model),
                {},
            ),
            (
                "FP8 weights + FP8 KV (`wk`)",
                one(m8, "m8_perplexity", model=model),
                one(m8, "m8_needle", model=model),
                {
                    t: 100 * v["score"]
                    for t, v in one(m8, "m8_tasks", model=model).get("scores", {}).items()
                    if v.get("score") is not None
                },
            ),
        ]
        for name, ppl, needle, scores in stacks:
            rows.append(
                [
                    model.split("/")[-1],
                    name,
                    _f(ppl.get("perplexity"), 2),
                    _f(100 * needle["pass_rate"], 0, "%") if "pass_rate" in needle else DASH,
                    *(_f(scores.get(task), 1) for task in ("gsm8k", "mmlu", "humaneval")),
                ]
            )
    headers = [
        "Model",
        "Stack",
        "Perplexity",
        "Needle recall",
        "GSM8K (%)",
        "MMLU (%)",
        "HumanEval pass@1 (%)",
    ]
    return markdown_table(headers, rows)


def perplexities(m8: Records, m5: Records, m4: Records, model: str) -> dict[str, float]:
    """vLLM's WikiText-2 perplexity per lossy part of a stack: "" (BF16), "w", "k", "wk", "a" (INT4)."""

    def last(records: Records, experiment: str, **match: Any) -> float | None:
        found = [m for m in _sorted(records, experiment) if all(m.get(k) == v for k, v in match.items())]
        return found[-1].get("perplexity") if found else None

    values = {
        "": last(m4, "m4_vllm_perplexity", model=model, format="bf16"),
        "w": last(m4, "m4_vllm_perplexity", model=model, format="fp8"),
        "a": last(m4, "m4_vllm_perplexity", model=model, format="awq"),
        "k": last(m5, "m5_vllm_perplexity", model=model),
        "wk": last(m8, "m8_perplexity", model=model),
    }
    return {name: value for name, value in values.items() if value is not None}


def kernel_projection(
    m8: Records, m7: Records, configs: dict[str, Any], model: str
) -> dict[str, float] | None:
    """What M7's kernel 2 would give vLLM on the long workload, from measured pieces. A projection.

    The `wk` server's step at 32k tokens minus its step at short context is the time FlashInfer spends
    reading the FP8 cache. Replace that with kernel 2's measured time on INT4 codes, once per layer:

        projected step = wk's short-context step + layers × kernel 2's layer time at 32,768 tokens

    Neither the kernel nor an INT4 cache exists in vLLM, and the kernel verifies one token per sequence, so
    this is for the stack without speculation.
    """
    from fastserve.report.m7 import att

    long, short = serving(m8, model, "wk", "long_32k"), serving(m8, model, "wk", "m8_latency")
    layer_ms = att(m7, 1, 32768, "triton-int4")
    if not long or not short or layer_ms is None:
        return None
    step = long["summary"]["tpot_ms"]["p50"]
    projected = short["summary"]["tpot_ms"]["p50"] + configs[model].num_layers * layer_ms
    ttft_s = long["summary"]["ttft_ms"]["p50"] / 1e3
    output = measured_load(long, None).output_len
    return {
        "fp8_step_ms": step,
        "fp8_attention_ms": step - short["summary"]["tpot_ms"]["p50"],
        "kernel_attention_ms": configs[model].num_layers * layer_ms,
        "step_ms": projected,
        "tok_s": output / (ttft_s + output * projected / 1e3),
        "measured_tok_s": long["summary"]["output_throughput"],
    }


def kernel_projection_table(m8: Records, m7: Records, configs: dict[str, Any]) -> str:
    rows = []
    for model in (SMALL, LARGE):
        x = kernel_projection(m8, m7, configs, model)
        if x:
            rows.append(
                [
                    model.split("/")[-1],
                    _f(x["fp8_step_ms"], 1),
                    _f(x["fp8_attention_ms"], 1),
                    _f(x["kernel_attention_ms"], 1),
                    _f(x["step_ms"], 1),
                    _f(x["fp8_step_ms"] / x["step_ms"], 2, "×"),
                ]
            )
    headers = [
        "Model",
        "Measured step, FP8 KV (ms)",
        "of which the KV read (ms)",
        "Kernel 2 on INT4 codes, all layers (ms)",
        "Projected step (ms)",
        "Projected gain",
    ]
    return markdown_table(headers, rows)


def cost_table(m8: Records, dollars_per_hour: float) -> str:
    """The final cost table: $ per 1M output tokens for stock BF16, the full stack, and the best stack."""
    rows = []
    for model in (SMALL, LARGE):
        for workload in WORKLOADS:
            base, full = tok_s(m8, model, BASE, workload), tok_s(m8, model, FULL, workload)
            if not base:
                continue
            top = best(m8, model, workload, allow_int4=False)
            rows.append(
                [
                    model.split("/")[-1],
                    WORKLOAD_LABELS[workload],
                    _f(dollars(base, dollars_per_hour), 2),
                    _f(dollars(full, dollars_per_hour), 2),
                    _f(full / base, 2, "×") if full else DASH,
                    f"`{top[0]}`" if top else DASH,
                    _f(dollars(top[1], dollars_per_hour), 2) if top else DASH,
                    _f(top[1] / base, 2, "×") if top else DASH,
                ]
            )
    headers = [
        "Model",
        "Workload",
        "Stock BF16 ($ per 1M)",
        "Full stack `wkps` ($ per 1M)",
        "Cheaper by",
        "Best measured stack",
        "Its $ per 1M",
        "Cheaper by",
    ]
    return markdown_table(headers, rows)


# ---- the best stack per workload, and why the full stack is not it -----------------------------------------

DEPLOYABLE = "wakps"  # the letters that are techniques; f and g exist only as controls


def candidates(m8: Records, model: str, allow_int4: bool = True) -> list[str]:
    """The servers a deployment could choose from: no repeats, no controls."""
    allowed = set(DEPLOYABLE) - (set() if allow_int4 else {"a"})
    return [
        label
        for label in labels_run(m8, model)
        if not label.endswith("-r2") and set(letters(label)) <= allowed
    ]


def best(m8: Records, model: str, workload: str, allow_int4: bool = True) -> tuple[str, float] | None:
    """(label, tokens/s) of the fastest measured candidate on one workload."""
    rates = {label: tok_s(m8, model, label, workload) for label in candidates(m8, model, allow_int4)}
    rates = {label: rate for label, rate in rates.items() if rate}
    if not rates:
        return None
    label = max(rates, key=rates.get)
    return label, rates[label]


def best_table(m8: Records, model: str, perplexity: dict[str, float] | None = None) -> str:
    """Per workload: the full stack against the best measured stack, with and without INT4 weights.

    `perplexity` (lossy letters → vLLM's WikiText-2 perplexity) adds what the best FP8-class stack costs in
    quality: speed is never reported without it.
    """
    rows = []
    for workload in WORKLOADS:
        base = tok_s(m8, model, BASE, workload)
        if not base:
            continue
        fp8, int4 = best(m8, model, workload, allow_int4=False), best(m8, model, workload)
        lossy = "".join(x for x in letters(fp8[0]) if x in "wk") if fp8 else ""
        change = None
        if perplexity and lossy in perplexity and "" in perplexity:
            change = 100 * (perplexity[lossy] / perplexity[""] - 1)
        rows.append(
            [
                WORKLOAD_LABELS[workload],
                _f(base, 0),
                _f(speedup(m8, model, FULL, workload), 2, "×"),
                f"`{fp8[0]}`" if fp8 else DASH,
                _f(fp8[1] / base, 2, "×") if fp8 else DASH,
                f"{change:+.1f}%" if change is not None else DASH,
                f"`{int4[0]}`" if int4 and "a" in letters(int4[0]) else DASH,
                _f(int4[1] / base, 2, "×") if int4 and "a" in letters(int4[0]) else DASH,
            ]
        )
    headers = [
        "Workload",
        "Stock BF16 (tokens/s)",
        "Full stack `wkps`",
        "Best measured stack",
        "Its speedup",
        "Its perplexity vs BF16",
        "Best with INT4 weights, if faster",
        "Its speedup",
    ]
    return markdown_table(headers, rows)


def graphs_of(m8: Records, model: str, label: str) -> dict[str, Any] | None:
    """What vLLM said at this server's startup about CUDA graphs (the newest record that has it)."""
    found = None
    for experiment in ("m8_graphs", "server_start"):
        for m in _sorted(m8, experiment):
            if (m["model"], m["label"]) == (model, label) and m.get("cuda_graphs"):
                found = m
    return found


def graph_mode(start: dict[str, Any] | None) -> str:
    """ "full" if a full graph of the decode pass was captured, else "piecewise"."""
    if not start:
        return DASH
    return "full" if "full" in start["cuda_graphs"]["captured"] else "piecewise"


def pass_ms(m8: Records, model: str, label: str, workload: str = "m8_latency") -> float | None:
    """Milliseconds per target pass: time per token × tokens kept per pass (1 without speculation)."""
    m = serving(m8, model, label, workload)
    if not m:
        return None
    kept = tokens_per_pass(m) if "s" in letters(label) else 1.0
    return m["summary"]["tpot_ms"]["p50"] * kept if kept else None


def collision_table(m8: Records, model: str, labels: list[str] | None = None) -> str:
    """The servers that separate the causes of the FP8-KV and speculation collision, at one user."""
    names = {
        BASE: "stock",
        "g": "stock, piecewise graphs only (control)",
        "k": "FP8 KV",
        "s": "speculation",
        "sg": "speculation, piecewise graphs only (control)",
        "fs": "speculation on FlashInfer, BF16 cache (control)",
        "ks": "speculation + FP8 KV",
    }
    rows = []
    for label in labels or list(names):
        m = serving(m8, model, label, "m8_latency")
        if not m:
            continue
        start = graphs_of(m8, model, label)
        rows.append(
            [
                f"`{label}`",
                names.get(label, label),
                (start or {}).get("attention_backend") or DASH,
                graph_mode(start),
                _f(tok_s(m8, model, label, "m8_latency"), 0),
                _f(tokens_per_pass(m) if "s" in letters(label) else 1.0, 2),
                _f(pass_ms(m8, model, label), 1),
                _f(power(m8, model, label, "m8_latency"), 0),
                _f(tok_s(m8, model, label, "spec_mixed"), 0),
            ]
        )
    headers = [
        "Server",
        "What it is",
        "Attention backend",
        "Decode pass in a CUDA graph",
        "Tokens/s, 1 user",
        "Tokens per pass",
        "ms per pass",
        "GPU power (W)",
        "Tokens/s, 64 users",
    ]
    return markdown_table(headers, rows)


def fallback_message(m8: Records, model: str, label: str) -> str | None:
    """vLLM's own words when it gave up the full graph on this server."""
    start = graphs_of(m8, model, label)
    return start["cuda_graphs"].get("fallback") if start else None


def control_observables(m8: Records) -> dict[str, float | None]:
    """The measured value of every quantity in benchmarks/predictions/m8_controls.json."""

    def ratio(label: str, over: str, workload: str) -> float | None:
        return speedup(m8, LARGE, label, workload, over)

    return {
        "control_fs_vs_ks_latency": ratio("fs", "ks", "m8_latency"),
        "control_fs_latency": ratio("fs", "s", "m8_latency"),
        "control_sg_latency": ratio("sg", "s", "m8_latency"),
        "control_g_latency": ratio("g", BASE, "m8_latency"),
        "control_g_busy": ratio("g", BASE, "spec_mixed"),
        "control_fs_busy": ratio("fs", "s", "spec_mixed"),
        "control_aps_latency": ratio("aps", BASE, "m8_latency"),
        "control_aps_multi_turn": ratio("aps", BASE, "multi_turn"),
    }


# ---- predictions -------------------------------------------------------------------------------------------


def m8_observables(
    m8: Records, frozen_file: dict[str, Any], m4: Records, m2_quality: Records
) -> dict[str, float | None]:
    """The measured value of every quantity in benchmarks/predictions/m8.json."""
    from fastserve.report import m4 as m4_report

    def sp(label: str, workload: str, model: str = LARGE, over: str = BASE) -> float | None:
        return speedup(m8, model, label, workload, over)

    def ratio(a: float | None, b: float | None) -> float | None:
        return a / b if a and b else None

    def gain_share() -> float | None:
        w, wf, wk = (sp(label, "capacity") for label in ("w", "wf", "wk"))
        return (wf - w) / (wk - w) if w and wf and wk and wk != w else None

    errors = prediction_errors(m8, frozen_file)
    plain = [r["error"] for r in errors if "s" not in letters(r["label"])]
    repeats = [abs(diff) for model in (LARGE, SMALL) for _, _, diff in repeat_differences(m8, model)]
    kept = serving(m8, LARGE, "s", "capacity")

    def one(experiment: str, model: str) -> dict[str, Any]:
        found = [m for m in _sorted(m8, experiment) if m["model"] == model]
        return found[-1] if found else {}

    base_ppl = [m for m in _sorted(m4, "m4_vllm_perplexity") if (m["model"], m["format"]) == (LARGE, "bf16")]
    gsm8k = (one("m8_tasks", LARGE).get("scores", {}).get("gsm8k") or {}).get("score")
    base_gsm8k = m4_report.tasks(m4, m2_quality, LARGE, "bf16").get("gsm8k")
    return {
        "full_latency": sp(FULL, "m8_latency"),
        "full_busy": sp(FULL, "spec_mixed"),
        "full_capacity": sp(FULL, "capacity"),
        "full_multi_turn": sp(FULL, "multi_turn"),
        "full_long": sp(FULL, "long_32k"),
        "int4_latency": sp("akps", "m8_latency", over=FULL),
        "int4_busy": sp("akps", "spec_mixed", over=FULL),
        "interaction_ws_latency": interaction(m8, LARGE, "w", "s", "m8_latency"),
        "interaction_wk_busy": interaction(m8, LARGE, "w", "k", "spec_mixed"),
        "interaction_ks_capacity": interaction(m8, LARGE, "k", "s", "capacity"),
        "interaction_ps_multi_turn": interaction(m8, LARGE, "p", "s", "multi_turn"),
        "interaction_wp_multi_turn": interaction(m8, LARGE, "w", "p", "multi_turn"),
        "interaction_kp_multi_turn": interaction(m8, LARGE, "k", "p", "multi_turn"),
        "loo_s_latency": sp(FULL, "m8_latency", over=without(FULL, "s")),
        "loo_w_latency": sp(FULL, "m8_latency", over=without(FULL, "w")),
        "loo_k_capacity": sp(FULL, "capacity", over=without(FULL, "k")),
        "loo_p_multi_turn": sp(FULL, "multi_turn", over=without(FULL, "p")),
        "prefix_without_prefixes": sp("wp", "m8_latency", over="w"),
        "flashinfer_share_capacity": gain_share(),
        "eagle_kept_capacity": tokens_per_pass(kept) if kept else None,
        "model_median_error": _median_abs([r["error"] for r in errors]),
        "model_within_15": sum(abs(r["error"]) <= 0.15 for r in errors) / len(errors) if errors else None,
        "model_median_error_plain": _median_abs(plain),
        "repeat_difference": max(repeats) if repeats else None,
        "energy_full_latency": ratio(
            tokens_per_joule(m8, LARGE, FULL, "m8_latency"), tokens_per_joule(m8, LARGE, BASE, "m8_latency")
        ),
        "power_busy_base": power(m8, LARGE, BASE, "spec_mixed"),
        "small_full_latency": sp(FULL, "m8_latency", SMALL),
        "small_full_busy": sp(FULL, "spec_mixed", SMALL),
        "small_full_multi_turn": sp(FULL, "multi_turn", SMALL),
        "quality_perplexity": ratio(
            one("m8_perplexity", LARGE).get("perplexity"), base_ppl[-1]["perplexity"] if base_ppl else None
        ),
        "quality_needle": one("m8_needle", LARGE).get("pass_rate"),
        "quality_gsm8k": gsm8k - base_gsm8k / 100 if gsm8k is not None and base_gsm8k is not None else None,
    }
