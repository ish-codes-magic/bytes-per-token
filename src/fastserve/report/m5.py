"""M5 analysis: KV policies in nanoserve (KL, needle), FP8 KV and prefix caching in vLLM. Stdlib only.

Records come from results/raw/m5_kv.jsonl. Baselines from earlier milestones: M2's BF16 needle grid
(results/raw/m2_quality.jsonl) and M4's BF16 vLLM perplexity and FP8-weight servers (m4_production.jsonl).
"""

from __future__ import annotations

from typing import Any

from fastserve.kv.sizing import KVSpec, TensorQuant, bytes_per_token, max_tokens
from fastserve.report.m2 import plateau
from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]
SMALL, LARGE = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"
DASH = "—"
KV_LABELS = {
    "bf16kv": "BF16 KV",
    "bf16kv-flashinfer": "BF16 KV, FlashInfer",
    "fp8kv": "FP8 KV",
    "fp8w-fp8kv": "FP8 weights + FP8 KV",
}
POLICY_LABELS = {
    "bf16": "BF16 (control)",
    "fp8": "FP8 E4M3, scale 1.0",
    "int8": "INT8, per token",
    "int8-kivi": "INT8, keys per channel",
    "int4-token": "INT4, per token",
    "int4-token-rot": "INT4, rotated keys",
    "int4-kivi": "INT4 KIVI (keys per channel)",
    "int2-kivi": "INT2 KIVI",
    "streaming": "StreamingLLM, 1,024 kept",
}


def _newest(records: Records, experiment: str) -> list[dict[str, Any]]:
    return [
        r["metrics"] for r in sorted(records, key=lambda r: r["timestamp"]) if r["experiment"] == experiment
    ]


def _one(records: Records, experiment: str, **match: Any) -> dict[str, Any] | None:
    found = [m for m in _newest(records, experiment) if all(m.get(k) == v for k, v in match.items())]
    return found[-1] if found else None


def _f(x: float | None, digits: int = 0, suffix: str = "") -> str:
    return DASH if x is None else f"{x:,.{digits}f}{suffix}"


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


# ---- vLLM records -----------------------------------------------------------------------------------------


def serving(records: Records, model: str, label: str, workload: str) -> dict[str, Any] | None:
    return _one(records, "serving", model=model, server=label, workload=workload)


def start(records: Records, model: str, label: str) -> dict[str, Any] | None:
    return _one(records, "server_start", model=model, label=label)


def held(records: Records, model: str, label: str, workload: str) -> dict[str, float] | None:
    """Rates while requests were queueing: the server running as many sequences as it could."""
    m = serving(records, model, label, workload)
    return plateau(m) if m and "server_timeline" in m else None


def summary_stat(records: Records, model: str, label: str, workload: str, metric: str) -> float | None:
    m = serving(records, model, label, workload)
    if not m:
        return None
    value = m["summary"].get(metric)
    return value.get("p50") if isinstance(value, dict) else value


def cached_share(records: Records, model: str, label: str, workload: str) -> float | None:
    """Prompt tokens vLLM served from its prefix cache ÷ all prompt tokens (per-request usage stats)."""
    m = serving(records, model, label, workload)
    if not m:
        return None
    cols = {c: i for i, c in enumerate(m["requests"]["columns"])}
    if "cached_tokens" not in cols:
        return None
    rows = m["requests"]["rows"]
    prompt = sum(r[cols["prompt_len"]] for r in rows)
    return sum(r[cols["cached_tokens"]] or 0 for r in rows) / prompt if prompt else None


# ---- nanoserve records ------------------------------------------------------------------------------------


def kl(records: Records, model: str, policy: str) -> dict[str, Any] | None:
    return _one(records, "m5_kv_kl", model=model, policy=policy)


def needle(records: Records, model: str, policy: str) -> dict[str, Any] | None:
    return _one(records, "m5_kv_needle", model=model, policy=policy)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def m5_observables(m5: Records, m4: Records) -> dict[str, float | None]:
    """The measured value behind each prediction in benchmarks/predictions/m5.json."""

    def tokens(model: str, label: str) -> float | None:
        return (start(m5, model, label) or {}).get("kv_cache_tokens")

    def tput(model: str, label: str, workload: str) -> float | None:
        p = held(m5, model, label, workload)
        return p["output_tok_s"] if p else summary_stat(m5, model, label, workload, "output_throughput")

    def kl_of(model: str, policy: str) -> float | None:
        m = kl(m5, model, policy)
        return m["mean_kl"] if m else None

    def needle_of(policy: str) -> float | None:
        m = needle(m5, SMALL, policy)
        return m["pass_rate"] if m else None

    stats = _one(m5, "m5_kv_stats", model=SMALL)
    vllm_needle = _one(m5, "m5_vllm_needle", model=SMALL)
    vllm_ppl = _one(m5, "m5_vllm_perplexity", model=SMALL)
    bf16_ppl = _one(m4, "m4_vllm_perplexity", model=SMALL, format="bf16")
    running = [held(m5, SMALL, label, "capacity") for label in ("fp8kv", "bf16kv")]
    return {
        "kv_tokens_fp8_small": _ratio(tokens(SMALL, "fp8kv"), tokens(SMALL, "bf16kv")),
        "kv_tokens_fp8_large": _ratio(tokens(LARGE, "fp8kv"), tokens(LARGE, "bf16kv")),
        "capacity_running_fp8_small": _ratio(*(p["running"] if p else None for p in running)),
        "capacity_tput_fp8_small": _ratio(
            tput(SMALL, "fp8kv", "capacity"), tput(SMALL, "bf16kv", "capacity")
        ),
        "capacity_tput_fp8_large": _ratio(
            tput(LARGE, "fp8kv", "capacity"), tput(LARGE, "bf16kv", "capacity")
        ),
        "sat_tput_fp8_small": _ratio(tput(SMALL, "fp8kv", "saturation"), tput(SMALL, "bf16kv", "saturation")),
        "sat_tput_fp8_large": _ratio(tput(LARGE, "fp8kv", "saturation"), tput(LARGE, "bf16kv", "saturation")),
        "sat_flashinfer_small": _ratio(
            tput(SMALL, "bf16kv-flashinfer", "saturation"), tput(SMALL, "bf16kv", "saturation")
        ),
        "tpot32k_fp8_small": _ratio(
            summary_stat(m5, SMALL, "bf16kv", "long_32k", "tpot_ms"),
            summary_stat(m5, SMALL, "fp8kv", "long_32k", "tpot_ms"),
        ),
        "tpot32k_fp8_large": _ratio(
            summary_stat(m5, LARGE, "bf16kv", "long_32k", "tpot_ms"),
            summary_stat(m5, LARGE, "fp8kv", "long_32k", "tpot_ms"),
        ),
        "ttft32k_fp8_small": _ratio(
            summary_stat(m5, SMALL, "fp8kv", "long_32k", "ttft_ms"),
            summary_stat(m5, SMALL, "bf16kv", "long_32k", "ttft_ms"),
        ),
        "kl_fp8": kl_of(SMALL, "fp8"),
        "kl_int8": kl_of(SMALL, "int8"),
        "kl_int4_token": kl_of(SMALL, "int4-token"),
        "kl_rot_over_token": _ratio(kl_of(SMALL, "int4-token-rot"), kl_of(SMALL, "int4-token")),
        "kl_kivi_over_token": _ratio(kl_of(SMALL, "int4-kivi"), kl_of(SMALL, "int4-token")),
        "kl_int2_kivi": kl_of(SMALL, "int2-kivi"),
        "kl_streaming": kl_of(SMALL, "streaming"),
        "kl_kivi_large_over_small": _ratio(kl_of(LARGE, "int4-kivi"), kl_of(SMALL, "int4-kivi")),
        "key_over_value_outliers": _ratio(_median(stats["key_ratio"]), _median(stats["value_ratio"]))
        if stats
        else None,
        "needle_bf16_nanoserve": needle_of("bf16"),
        "needle_fp8": needle_of("fp8"),
        "needle_int4_token": needle_of("int4-token"),
        "needle_int4_kivi": needle_of("int4-kivi"),
        "needle_streaming": needle_of("streaming"),
        "needle_vllm_fp8_small": vllm_needle["pass_rate"] if vllm_needle else None,
        "ppl_vllm_fp8_small": _ratio((vllm_ppl or {}).get("perplexity"), (bf16_ppl or {}).get("perplexity")),
        "prefix_cached_share": cached_share(m5, SMALL, "prefix-on", "multi_turn"),
        "prefix_ttft_small": _ratio(
            summary_stat(m5, SMALL, "prefix-on", "multi_turn", "ttft_ms"),
            summary_stat(m5, SMALL, "prefix-off", "multi_turn", "ttft_ms"),
        ),
        "prefix_tput_small": _ratio(
            summary_stat(m5, SMALL, "prefix-on", "multi_turn", "output_throughput"),
            summary_stat(m5, SMALL, "prefix-off", "multi_turn", "output_throughput"),
        ),
        "prefix_tput_large": _ratio(
            summary_stat(m5, LARGE, "prefix-on", "multi_turn", "output_throughput"),
            summary_stat(m5, LARGE, "prefix-off", "multi_turn", "output_throughput"),
        ),
    }


# ---- tables -----------------------------------------------------------------------------------------------


def policy_spec(policy: str, policies: list[dict[str, Any]]) -> KVSpec:
    """The KVSpec of a configured policy (stdlib copy of experiments.m5.kv_spec, which needs torch)."""
    entry = next(p for p in policies if p["name"] == policy)

    def tensor(cfg: dict[str, Any] | None) -> TensorQuant | None:
        return None if cfg is None else TensorQuant(**cfg)

    return KVSpec(
        policy,
        keys=tensor(entry.get("keys")),
        values=tensor(entry.get("values")),
        rotate_keys=entry.get("rotate_keys", False),
        sinks=entry.get("sinks"),
        window=entry.get("window"),
    )


def policy_table(
    m5: Records, policies: list[dict[str, Any]], cfg: Any, kv_bytes: float, context: int = 32000
) -> str:
    """Per KV policy: storage cost, what fits in the L4's KV budget, and quality (KL on both models,
    needle recall)."""
    rows = []
    for p in policies:
        spec = policy_spec(p["name"], policies)
        fits = int(kv_bytes // (spec.kept_tokens(context) * bytes_per_token(cfg, spec)))
        small, large, hay = kl(m5, SMALL, p["name"]), kl(m5, LARGE, p["name"]), needle(m5, SMALL, p["name"])
        rows.append(
            [
                POLICY_LABELS.get(p["name"], p["name"]),
                _f(spec.bits_per_element(), 2),
                _f(bytes_per_token(cfg, spec) / 1024, 1),
                _f(fits, 0),
                _f((small or {}).get("mean_kl"), 4),
                _f((large or {}).get("mean_kl"), 4),
                _f(100 * small["top1_agreement"], 1, "%") if small else DASH,
                _f(100 * hay["pass_rate"], 0, "%") if hay else DASH,
            ]
        )
    headers = [
        "KV policy",
        "Bits per element",
        "KiB per token",
        f"{context // 1000}k-token sequences that fit",
        "KL, Qwen3-0.6B",
        "KL, Qwen3-1.7B",
        "Top-1 agreement, 0.6B",
        "Needle pass rate, 0.6B",
    ]
    return markdown_table(headers, rows)


def kv_serving_table(m5: Records, dollars_per_hour: float = 0.80) -> str:
    """vLLM per KV format: capacity, throughput when full, long-context latency, and cost."""
    rows = []
    for model in (SMALL, LARGE):
        for label in KV_LABELS:
            s = start(m5, model, label)
            if not s:
                continue
            cap, sat = held(m5, model, label, "capacity"), held(m5, model, label, "saturation")
            cost = dollars_per_hour / (sat["output_tok_s"] * 3600) * 1e6 if sat else None
            rows.append(
                [
                    model.split("/")[-1],
                    KV_LABELS[label],
                    s.get("attention_backend") or DASH,
                    _f(s.get("kv_cache_tokens"), 0),
                    _f(cap["running"], 0) if cap else DASH,
                    _f(cap["output_tok_s"], 0) if cap else DASH,
                    _f(sat["output_tok_s"], 0) if sat else DASH,
                    _f(summary_stat(m5, model, label, "long_32k", "tpot_ms"), 2),
                    _f(summary_stat(m5, model, label, "long_32k", "ttft_ms"), 0),
                    _f(cost, 3),
                ]
            )
    headers = [
        "Model",
        "Server",
        "Attention",
        "KV cache (tokens)",
        "Running, capacity workload",
        "Tokens/s, capacity",
        "Tokens/s, saturated",
        "TPOT at 32k (ms)",
        "TTFT at 32k (ms)",
        "$ / 1M tokens, saturated",
    ]
    return markdown_table(headers, rows)


def prefix_table(m5: Records, dollars_per_hour: float = 0.80) -> str:
    rows = []
    for model in (SMALL, LARGE):
        for workload in ("multi_turn", "shared_prefix"):
            for label, name in (("prefix-off", "off"), ("prefix-on", "on")):
                if not serving(m5, model, label, workload):
                    continue
                tok_s = summary_stat(m5, model, label, workload, "output_throughput")
                rows.append(
                    [
                        model.split("/")[-1],
                        workload.replace("_", "-"),
                        name,
                        _f(100 * (cached_share(m5, model, label, workload) or 0), 1, "%"),
                        _f(summary_stat(m5, model, label, workload, "ttft_ms"), 0),
                        _f(summary_stat(m5, model, label, workload, "tpot_ms"), 2),
                        _f(tok_s, 0),
                        _f(dollars_per_hour / (tok_s * 3600) * 1e6 if tok_s else None, 3),
                    ]
                )
    headers = [
        "Model",
        "Workload",
        "Prefix caching",
        "Prompt tokens cached",
        "TTFT p50 (ms)",
        "TPOT p50 (ms)",
        "Tokens/s",
        "$ / 1M tokens",
    ]
    return markdown_table(headers, rows)


def vllm_quality_table(m5: Records, m2_quality: Records, m4: Records) -> str:
    """vLLM with an FP8 KV cache against its BF16 baselines (M4's perplexity, M2's needle grid)."""
    rows = []
    for model in (SMALL, LARGE):
        base_ppl = (_one(m4, "m4_vllm_perplexity", model=model, format="bf16") or {}).get("perplexity")
        base_needle = (_one(m2_quality, "quality_needle", model=model) or {}).get("pass_rate")
        ppl = (_one(m5, "m5_vllm_perplexity", model=model) or {}).get("perplexity")
        hay = (_one(m5, "m5_vllm_needle", model=model) or {}).get("pass_rate")
        rows.append(
            [
                model.split("/")[-1],
                _f(base_ppl, 2),
                _f(ppl, 2),
                _f(100 * (_ratio(ppl, base_ppl) - 1), 2, "%") if _ratio(ppl, base_ppl) else DASH,
                _f(None if base_needle is None else 100 * base_needle, 0, "%"),
                _f(None if hay is None else 100 * hay, 0, "%"),
            ]
        )
    headers = [
        "Model",
        "Perplexity, BF16 KV",
        "Perplexity, FP8 KV",
        "Change",
        "Needle, BF16 KV",
        "Needle, FP8 KV",
    ]
    return markdown_table(headers, rows)


def capacity_budget(cfg: Any, kv_bytes: float, spec: KVSpec) -> int:
    """Tokens the L4's KV budget holds in a format (the sizing model behind the concurrency figure)."""
    return max_tokens(cfg, kv_bytes, spec)
