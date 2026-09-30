"""M2 analysis: serving records → the numbers and tables the docs use. Stdlib only."""

from __future__ import annotations

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
