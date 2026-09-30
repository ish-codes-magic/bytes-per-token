"""M2 analysis: serving records → the numbers and tables the docs use. Stdlib only."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from fastserve.hw.analysis import metrics_of
from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]
SMALL, LARGE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"


def runs(records: Records, *, model: str, workload: str, mode: str | None = None) -> list[dict[str, Any]]:
    """Serving load points for one model and workload (optionally only "open" or "closed" loads)."""
    return [
        m
        for m in metrics_of(records, "serving")
        if m["model"] == model and m["workload"] == workload and (mode is None or m["load"]["mode"] == mode)
    ]


def _p(summary: dict[str, Any], metric: str, stat: str = "p50") -> float | None:
    return (summary.get(metric) or {}).get(stat)


def peak_throughput(records: Records, model: str) -> float | None:
    rows = runs(records, model=model, workload="throughput")
    return max((m["summary"].get("output_throughput", 0.0) for m in rows), default=None)


def knee_rate(records: Records, model: str, *, threshold: float = 0.9) -> float | None:
    """The highest open-loop arrival rate at which at least `threshold` of requests met the SLO."""
    ok = [
        m["load"]["rate"]
        for m in runs(records, model=model, workload="throughput", mode="open")
        if m["summary"].get("goodput_fraction", 0.0) >= threshold
    ]
    return max(ok, default=None)


def chat(records: Records, model: str) -> dict[str, Any] | None:
    rows = runs(records, model=model, workload="chat")
    return rows[0]["summary"] if rows else None


def m2_observables(records: Records, *, nanoserve_b1_tok_s: float | None) -> dict[str, float | None]:
    """The quantities predicted before M2 (benchmarks/predictions/m2.json)."""
    small, large = chat(records, SMALL), chat(records, LARGE)
    small_tpot = _p(small, "tpot_ms") if small else None
    long32 = runs(records, model=SMALL, workload="long_32k")
    shared = runs(records, model=SMALL, workload="shared_prefix")
    peak_small, peak_large = peak_throughput(records, SMALL), peak_throughput(records, LARGE)
    return {
        "chat_tpot_ms": small_tpot,
        "chat_ttft_ms": _p(small, "ttft_ms") if small else None,
        "vllm_over_nanoserve": (1e3 / small_tpot) / nanoserve_b1_tok_s
        if small_tpot and nanoserve_b1_tok_s
        else None,
        "peak_tok_s": peak_small,
        "knee_rate": knee_rate(records, SMALL),
        "tpot_ratio_1_7b": _p(large, "tpot_ms") / small_tpot if large and small_tpot else None,
        "throughput_ratio_1_7b": peak_large / peak_small if peak_large and peak_small else None,
        "long32k_ttft_s": _p(long32[0]["summary"], "ttft_ms") / 1e3 if long32 else None,
        "long32k_tpot_ratio": _p(long32[0]["summary"], "tpot_ms") / small_tpot
        if long32 and small_tpot
        else None,
        "shared_prefix_peak_rate": max(
            (m["summary"].get("request_throughput", 0.0) for m in shared), default=None
        ),
    }


def _fmt(x: float | None, digits: int = 0) -> str:
    return "—" if x is None else f"{x:,.{digits}f}"


def summary_table(records: Records) -> str:
    starts = {m["model"]: m for m in metrics_of(records, "server_start")}
    rows = []
    for model in (SMALL, LARGE):
        c = chat(records, model)
        peak = peak_throughput(records, model)
        peak_row = max(
            runs(records, model=model, workload="throughput"),
            key=lambda m: m["summary"].get("output_throughput", 0),
            default=None,
        )
        rows.append(
            [
                model.split("/")[-1],
                _fmt(_p(c, "ttft_ms") if c else None, 1),
                _fmt(_p(c, "tpot_ms") if c else None, 2),
                _fmt(1e3 / _p(c, "tpot_ms") if c and _p(c, "tpot_ms") else None),
                _fmt(peak),
                _fmt(knee_rate(records, model)),
                _fmt(peak_row["summary"].get("dollars_per_million_output_tokens") if peak_row else None, 3),
                _fmt(starts.get(model, {}).get("kv_cache_tokens")),
            ]
        )
    headers = [
        "Model",
        "Chat TTFT p50 (ms)",
        "Chat TPOT p50 (ms)",
        "Single-stream tokens/s",
        "Peak tokens/s",
        "Knee (req/s)",
        "$ / 1M tokens at peak",
        "KV cache (tokens)",
    ]
    return markdown_table(headers, rows)


def load_table(records: Records, model: str, workload: str = "throughput") -> str:
    rows = []
    for m in runs(records, model=model, workload=workload):
        s, load = m["summary"], m["load"]
        offered = f"{load['rate']} req/s" if load["mode"] == "open" else f"{load['concurrency']} users"
        rows.append(
            [
                offered,
                _fmt(s.get("request_throughput"), 1),
                _fmt(s.get("output_throughput")),
                _fmt(_p(s, "ttft_ms")),
                _fmt(_p(s, "ttft_ms", "p99")),
                _fmt(_p(s, "tpot_ms"), 1),
                _fmt(_p(s, "tpot_ms", "p99"), 1),
                f"{100 * s.get('goodput_fraction', 0):.0f}%",
                _fmt(s.get("dollars_per_million_output_tokens"), 3),
            ]
        )
    headers = [
        "Offered load",
        "Achieved req/s",
        "Output tok/s",
        "TTFT p50 (ms)",
        "TTFT p99 (ms)",
        "TPOT p50 (ms)",
        "TPOT p99 (ms)",
        "Within SLO",
        "$ / 1M tokens",
    ]
    return markdown_table(headers, rows)


def long_context_table(records: Records) -> str:
    rows = []
    for model in (SMALL, LARGE):
        for workload in ("chat", "long_8k", "long_16k", "long_32k"):
            for m in runs(records, model=model, workload=workload):
                s = m["summary"]
                prompt = "~100 (chat)" if workload == "chat" else workload.removeprefix("long_")
                rows.append([model.split("/")[-1], prompt, _fmt(_p(s, "ttft_ms")), _fmt(_p(s, "tpot_ms"), 2)])
    return markdown_table(["Model", "Prompt tokens", "TTFT p50 (ms)", "TPOT p50 (ms)"], rows)


# -- quality baselines ----------------------------------------------------------------------------


def quality_by_model(records: Records) -> dict[str, dict[str, Any]]:
    """{model: {"perplexity", "needle" (pass %), "needle_cells", "gsm8k", "mmlu", "humaneval" (%)}}."""
    out: dict[str, dict[str, Any]] = {}
    for m in metrics_of(records, "quality_perplexity"):
        out.setdefault(m["model"], {})["perplexity"] = m["perplexity_ref"]
    for m in metrics_of(records, "quality_needle"):
        entry = out.setdefault(m["model"], {})
        entry["needle"], entry["needle_cells"] = 100 * m["pass_rate"], m["cells"]
    for m in metrics_of(records, "quality_tasks"):
        for task, score in m["scores"].items():
            value = score.get("score")
            out.setdefault(m["model"], {})[task] = None if value is None else 100 * value
    return out


def quality_observables(records: Records) -> dict[str, float | None]:
    q = quality_by_model(records)
    small, large = q.get(SMALL, {}), q.get(LARGE, {})
    obs = {}
    for key, field in (
        ("ppl", "perplexity"),
        ("gsm8k", "gsm8k"),
        ("mmlu", "mmlu"),
        ("humaneval", "humaneval"),
        ("needle", "needle"),
    ):
        obs[f"{key}_small"], obs[f"{key}_large"] = small.get(field), large.get(field)
    return obs


def quality_table(records: Records) -> str:
    rows = []
    for model, q in quality_by_model(records).items():
        rows.append(
            [
                model.split("/")[-1],
                _fmt(q.get("perplexity"), 2),
                _fmt(q.get("gsm8k"), 1),
                _fmt(q.get("mmlu"), 1),
                _fmt(q.get("humaneval"), 1),
                _fmt(q.get("needle"), 0),
            ]
        )
    headers = [
        "Model",
        "WikiText-2 perplexity",
        "GSM8K (%)",
        "MMLU (%)",
        "HumanEval pass@1 (%)",
        "Needle (%)",
    ]
    return markdown_table(headers, sorted(rows))


def newest_per_model(records: Records) -> Records:
    """Results gathered from separate containers (one run id each): the newest per (experiment, model)."""
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in records:
        key = (r["experiment"], r["metrics"].get("model", ""))
        if key not in newest or r["timestamp"] > newest[key]["timestamp"]:
            newest[key] = r
    return list(newest.values())


def offline_table(offline: Records, serving: Records) -> str:
    """vLLM's engine alone vs the same engine behind its HTTP server and our client."""
    rows = []
    for m in sorted(metrics_of(offline, "offline_throughput"), key=lambda m: m["model"]):
        served = peak_throughput(serving, m["model"])
        rows.append(
            [
                m["model"].split("/")[-1],
                _fmt(m["output_throughput"]),
                _fmt(served),
                f"{served / m['output_throughput']:.0%}" if served else "—",
            ]
        )
    headers = ["Model", "Engine alone (tokens/s)", "Served peak (tokens/s)", "Served / engine"]
    return markdown_table(headers, rows)


def prefill_flops(cfg: Any, length: int) -> float:
    """FLOPs to prefill `length` tokens.

    Every layer weight once per token, the LM head once (last token only), and causal attention: QKᵀ and PV
    over half of the length × length matrix, as FlashAttention computes it.
    """
    embed = cfg.vocab_size * cfg.hidden_size
    layer_params = cfg.num_params() - embed - (0 if cfg.tie_word_embeddings else embed)
    attention = 2 * cfg.num_heads * cfg.head_dim * length * length * cfg.num_layers
    return 2 * layer_params * length + 2 * embed + attention


def prefill_efficiency_table(records: Records, configs: dict[str, Any], peak_flops: float) -> str:
    """Single-user long-context prefill: how many TFLOP/s the GPU actually delivers (TTFT ≈ prefill time)."""
    rows = []
    for model, cfg in configs.items():
        for workload in ("long_8k", "long_16k", "long_32k"):
            for m in runs(records, model=model, workload=workload):
                table = m["requests"]
                length = round(
                    statistics.median(r[table["columns"].index("prompt_len")] for r in table["rows"])
                )
                seconds = _p(m["summary"], "ttft_ms") / 1e3
                flops = prefill_flops(cfg, length)
                rows.append(
                    [
                        model.split("/")[-1],
                        f"{length:,}",
                        _fmt(seconds * 1e3),
                        _fmt(flops / 1e12, 1),
                        _fmt(flops / seconds / 1e12, 1),
                        f"{flops / seconds / peak_flops:.0%}",
                    ]
                )
    headers = [
        "Model",
        "Prompt tokens",
        "TTFT p50 (ms)",
        "Prefill TFLOP",
        "Effective TFLOP/s",
        "% of M0 BF16 peak",
    ]
    return markdown_table(headers, rows)


def _columns(table: dict[str, Any]) -> dict[str, list]:
    """A columnar {"columns", "rows"} table -> {column: values}."""
    return {name: [row[i] for row in table["rows"]] for i, name in enumerate(table["columns"])}


def mean_decoding(requests: dict[str, Any]) -> float:
    """Time-averaged number of requests between first token and finish, over the whole run (client side)."""
    c = _columns(requests)
    events = sorted([(t, 1) for t in c["first_token"]] + [(t, -1) for t in c["finished"]])
    area, running, last = 0.0, 0, min(c["sent"])
    for t, delta in events:
        area += running * (t - last)
        running, last = running + delta, t
    return area / (max(c["finished"]) - min(c["sent"]))


_TIMELINE_KEYS = ("t", "running", "waiting", "kv_usage", "prompt_tokens", "generation_tokens", "steps")


def _intervals(metrics: dict[str, Any]) -> list[tuple[dict, dict]]:
    """Consecutive pairs of complete samples from a load point's server timeline."""
    c = _columns(metrics["server_timeline"])
    values = zip(*(c[k] for k in _TIMELINE_KEYS), strict=True)
    return list(pairwise(dict(zip(_TIMELINE_KEYS, v, strict=True)) for v in values if None not in v))


def queued(a: dict, b: dict) -> bool:
    """Requests waited at both ends of the interval: the engine was as full as vLLM makes it."""
    return a["waiting"] > 0 and b["waiting"] > 0


def drained(a: dict, b: dict, min_running: int = 50) -> bool:
    """The queue is gone, so no new prompts arrive, but a big batch is still decoding."""
    return a["waiting"] == 0 and b["waiting"] == 0 and min(a["running"], b["running"]) >= min_running


def plateau(metrics: dict[str, Any], keep: Callable[[dict, dict], bool] = queued) -> dict[str, float] | None:
    """Rates and per-step averages over the timeline intervals `keep` selects (default: saturated ones).

    Saturated = requests are waiting: the engine already runs as many sequences as vLLM allows, so the rates
    measured there are its capacity. Rates are counter differences. The running count and KV usage are
    averaged per engine step: each interval's value is weighted by the steps it contains.
    """
    total = dict.fromkeys(("seconds", "steps", "prompt", "generated", "seq_steps", "kv_steps"), 0.0)
    for a, b in _intervals(metrics):
        if not keep(a, b):
            continue
        steps = b["steps"] - a["steps"]
        total["seconds"] += b["t"] - a["t"]
        total["steps"] += steps
        total["prompt"] += b["prompt_tokens"] - a["prompt_tokens"]
        total["generated"] += b["generation_tokens"] - a["generation_tokens"]
        total["seq_steps"] += steps * (a["running"] + b["running"]) / 2
        total["kv_steps"] += steps * (a["kv_usage"] + b["kv_usage"]) / 2
    seconds, steps = total["seconds"], total["steps"]
    if not seconds or not steps:
        return None
    return {
        "seconds": seconds,
        "running": total["seq_steps"] / steps,
        "kv_usage": total["kv_steps"] / steps,
        "output_tok_s": total["generated"] / seconds,
        "prompt_tok_s": total["prompt"] / seconds,
        "prompt_per_step": total["prompt"] / steps,
        "step_ms": seconds / steps * 1e3,
    }


def memory_bound_step_s(kv_tokens: float, cfg: Any, bandwidth: float) -> float:
    """The shortest possible decode step: stream the weights once plus every cached token's K and V."""
    return (2 * cfg.num_params() + kv_tokens * cfg.kv_bytes_per_token()) / bandwidth


def memory_efficiency(p: dict[str, float], kv_capacity: int, cfg: Any, bandwidth: float) -> float:
    """Time the average step would take just streaming its bytes ÷ the time it took.

    Memory time is linear in the cached tokens, so the per-step average KV usage gives its per-step average.
    """
    return memory_bound_step_s(p["kv_usage"] * kv_capacity, cfg, bandwidth) / (p["step_ms"] / 1e3)


def saturation_run(records: Records, model: str, workload: str) -> tuple[dict, dict] | None:
    """(load point, its saturated plateau) for one model and workload of the saturation experiment."""
    for m in runs(records, model=model, workload=workload):
        if "server_timeline" in m and (p := plateau(m)):
            return m, p
    return None


def kv_capacity(records: Records, model: str) -> int:
    """KV-cache size in tokens, as vLLM logged it when this run's server started."""
    return next(m for m in metrics_of(records, "server_start") if m["model"] == model)["kv_cache_tokens"]


def token_weighted_context(requests: dict[str, Any]) -> float:
    """The average context of a decoding sequence, predicted from request lengths alone.

    While a request decodes, its context grows from prompt to prompt + output, and it holds a place in the
    batch for every output token. So a running sequence's context averages
        Σ (out·prompt + out·(out − 1)/2) / Σ out
    Long requests stay longest, which pulls the average above that of a typical request.
    """
    c = _columns(requests)
    pairs = list(zip(c["prompt_len"], c["output_tokens"], strict=True))
    return sum(o * p + o * (o - 1) / 2 for p, o in pairs) / sum(o for _, o in pairs)


def saturation_table(saturation: Records, configs: dict[str, Any], bandwidth: float) -> str:
    """Held at saturation: what the engine sustains, against the ceiling set by streaming its bytes.

    The server's KV-cache usage gives the tokens cached across all running sequences, so the bytes each step
    must read are measured: weights + cached tokens × KV bytes per token. The ceiling is the output rate if
    every step took only the time to read them at the M0 bandwidth.
    """
    rows = []
    for model, cfg in configs.items():
        if not (found := saturation_run(saturation, model, "saturation")):
            continue
        m, p = found
        capacity = kv_capacity(saturation, model)
        efficiency = memory_efficiency(p, capacity, cfg, bandwidth)
        measured_context = p["kv_usage"] * capacity / p["running"]
        rows.append(
            [
                model.split("/")[-1],
                _fmt(p["seconds"]),
                _fmt(p["running"]),
                f"{_fmt(token_weighted_context(m['requests']))} / {_fmt(measured_context)}",
                f"{p['kv_usage']:.0%}",
                _fmt(p["output_tok_s"]),
                _fmt(p["output_tok_s"] / efficiency),
                f"{efficiency:.0%}",
            ]
        )
    headers = [
        "Model",
        "Saturated for (s)",
        "Running sequences",
        "Context per sequence: predicted / measured",
        "KV cache used",
        "Output tok/s",
        "Memory-bound ceiling (tok/s)",
        "Measured / ceiling",
    ]
    return markdown_table(headers, rows)


_ONE_USER = {"chat": "one user, chat", "long_8k": "one user, 8k prompt", "long_16k": "one user, 16k prompt"}
_ONE_USER["long_32k"] = "one user, 32k prompt"


def decode_efficiency_table(
    baseline: Records, saturation: Records, configs: dict[str, Any], bandwidth: float
) -> str:
    """How close each decode step gets to the time it takes to stream its bytes.

    One user: TPOT p50, with the context averaged over the answer (prompt + half the output). Saturated: the
    server timeline, once while requests wait (every step also prefills new prompts) and once after the
    queue drains (decode only).
    """
    rows = []

    def row(model: str, label: str, sequences: float, context: float, prompt: str, step_s: float) -> None:
        memory_s = memory_bound_step_s(sequences * context, configs[model], bandwidth)
        rows.append(
            [
                model.split("/")[-1],
                label,
                _fmt(sequences),
                _fmt(context),
                prompt,
                _fmt(memory_s * bandwidth / 1e9, 2),
                _fmt(memory_s * 1e3, 1),
                _fmt(step_s * 1e3, 1),
                f"{memory_s / step_s:.0%}",
            ]
        )

    for model in configs:
        for workload, label in _ONE_USER.items():
            for m in runs(baseline, model=model, workload=workload):
                c = _columns(m["requests"])
                pairs = zip(c["prompt_len"], c["output_tokens"], strict=True)
                context = statistics.fmean(p + (o - 1) / 2 for p, o in pairs)
                row(model, label, 1, context, "0", _p(m["summary"], "tpot_ms") / 1e3)
        if found := saturation_run(saturation, model, "saturation"):
            capacity = kv_capacity(saturation, model)
            for label, keep in (("saturated, queue waiting", queued), ("saturated, queue drained", drained)):
                if p := plateau(found[0], keep):
                    kv = p["kv_usage"] * capacity
                    row(
                        model,
                        label,
                        p["running"],
                        kv / p["running"],
                        _fmt(p["prompt_per_step"]),
                        p["step_ms"] / 1e3,
                    )
    headers = [
        "Model",
        "Decoding",
        "Sequences",
        "Context per sequence",
        "Prompt tokens per step",
        "Bytes per step (GB)",
        "Step at memory speed (ms)",
        "Measured step (ms)",
        "Memory efficiency",
    ]
    return markdown_table(headers, rows)


def peak_definition_table(baseline: Records, saturation: Records) -> str:
    """The sweep's "peak" (a whole-run average, ramp-up and drain included) vs the saturated plateau."""
    rows = []
    for model in (SMALL, LARGE):
        sweep = runs(baseline, model=model, workload="throughput")
        found = saturation_run(saturation, model, "saturation")
        if not sweep or not found:
            continue
        best = max(sweep, key=lambda m: m["summary"]["output_throughput"])
        load = best["load"]
        where = f"{load['rate']:g} req/s" if load["mode"] == "open" else f"{load['concurrency']} users"
        _, p = found
        rows.append(
            [
                model.split("/")[-1],
                f"{_fmt(best['summary']['output_throughput'])} ({where})",
                _fmt(mean_decoding(best["requests"])),
                _fmt(p["output_tok_s"]),
                _fmt(p["running"]),
            ]
        )
    headers = [
        "Model",
        "Sweep peak: whole-run average (tok/s)",
        "Average decoding in that run",
        "Saturated plateau (tok/s)",
        "Running on the plateau",
    ]
    return markdown_table(headers, rows)


def prefill_budget_table(saturation: Records) -> str:
    """Shared-prefix traffic at saturation: the per-step token budget, not compute, sets the batch size.

    Each engine step processes at most B tokens (vLLM's max_num_batched_tokens): one per running sequence,
    the rest spent on prompts. With P prompt tokens and O output tokens per request, a request enters every
    P / (B − R) steps and stays about O steps, so by Little's law R = O × (B − R) / P, i.e. R = B·O / (P + O).
    B is measured here: the prompt and output tokens the server processed per step while requests waited.
    """
    rows = []
    for model in (SMALL, LARGE):
        if not (found := saturation_run(saturation, model, "shared_prefix")):
            continue
        m, p = found
        c = _columns(m["requests"])
        prompt, output = statistics.fmean(c["prompt_len"]), statistics.fmean(c["output_tokens"])
        budget = (p["prompt_tok_s"] + p["output_tok_s"]) * p["step_ms"] / 1e3
        rows.append(
            [
                model.split("/")[-1],
                _fmt(budget),
                f"{prompt:,.0f} + {output:.0f}",
                _fmt(budget * output / (prompt + output)),
                _fmt(p["running"]),
                _fmt(p["step_ms"]),
                _fmt(p["output_tok_s"] / output, 1),
            ]
        )
    headers = [
        "Model",
        "Tokens per step",
        "Tokens per request (prompt + output)",
        "Running: Little's law",
        "Running: measured",
        "Step (ms)",
        "Requests/s",
    ]
    return markdown_table(headers, rows)
