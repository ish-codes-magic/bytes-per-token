"""M1 analysis: raw nanoserve records → the numbers and tables the docs use. Stdlib only."""

from __future__ import annotations

from typing import Any

from fastserve.engine.config import ModelConfig
from fastserve.hw.analysis import metrics_of, single
from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]


def _by(records: Records, experiment: str, key: str) -> dict[Any, dict[str, Any]]:
    return {m[key]: m for m in metrics_of(records, experiment)}


def _profile(records: Records, prefix: str) -> dict[str, Any] | None:
    return next((m for m in metrics_of(records, "kernel_profile") if m["label"].startswith(prefix)), None)


def m1_observables(records: Records) -> dict[str, float | None]:
    """The quantities predicted before M1 (benchmarks/predictions/m1.json)."""
    parity = [r for r in metrics_of(records, "hf_parity") if r.get("hf_attention", "eager") == "eager"]
    decode = _by(records, "decode_speed", "batch")
    prefill = _by(records, "prefill_speed", "length")
    decode_profile = _profile(records, "decode")
    tokens = sum(r["tokens"] for r in parity)

    def speedup(batch: int) -> float | None:
        if batch in decode and 1 in decode:
            return decode[batch]["tokens_per_s"] / decode[1]["tokens_per_s"]
        return None

    return {
        "parity_top1": 100 * sum(r["top1_agreement"] * r["tokens"] for r in parity) / tokens
        if tokens
        else None,
        "parity_max_abs": max((r["max_abs_diff"] for r in parity), default=None),
        "decode_kernels": decode_profile["kernels"] if decode_profile else None,
        "decode_b1_tok_s": decode[1]["tokens_per_s"] if 1 in decode else None,
        "decode_b1_gpu_busy_pct": (
            100 * decode_profile["kernel_ms"] / decode[1]["step_ms_median"]
            if decode_profile and 1 in decode
            else None
        ),
        "decode_b16_over_b1": speedup(16),
        "decode_b64_over_b1": speedup(64),
        "prefill_512_ms": prefill[512]["ms_median"] if 512 in prefill else None,
        "prefill_2048_ms": prefill[2048]["ms_median"] if 2048 in prefill else None,
    }


def parity_table(records: Records) -> str:
    rows = []
    for m in metrics_of(records, "hf_parity"):
        first_line = m["prompt"].splitlines()[0]
        truncated = len(first_line) > 40 or "\n" in m["prompt"]
        prompt = first_line[:40].replace("|", r"\|") + ("…" if truncated else "")
        rows.append(
            [
                m.get("hf_attention", "eager"),
                prompt,
                str(m["tokens"]),
                f"{100 * m['top1_agreement']:.1f}%",
                f"{m['max_abs_diff']:.3f}",
                f"{m['mean_abs_diff']:.4f}",
                f"{m['mean_kl_hf_to_ours']:.2e}",
            ]
        )
    headers = [
        "HF attention",
        "Prompt",
        "Tokens",
        "Top-1",
        r"Max \|Δlogit\|",
        r"Mean \|Δlogit\|",
        "KL(HF ‖ ours)",
    ]
    return markdown_table(headers, rows)


def speed_table(records: Records) -> str:
    decode = sorted(metrics_of(records, "decode_speed"), key=lambda m: m["batch"])
    base = decode[0]["tokens_per_s"] if decode else None
    rows = [
        [
            f"decode, batch {m['batch']}",
            f"{m['step_ms_median']:.1f}",
            f"{m['tokens_per_s']:,.0f}",
            f"{m['tokens_per_s'] / base:.1f}×" if base else "—",
        ]
        for m in decode
    ]
    for m in sorted(metrics_of(records, "prefill_speed"), key=lambda m: m["length"]):
        tokens_per_s = m["length"] / (m["ms_median"] / 1e3)
        rows.append([f"prefill, {m['length']} tokens", f"{m['ms_median']:.1f}", f"{tokens_per_s:,.0f}", "—"])
    return markdown_table(["Workload", "Time per step (ms)", "Tokens/s", "vs batch 1"], rows)


def profile_table(records: Records) -> str:
    decode = _by(records, "decode_speed", "batch")
    prefill = _by(records, "prefill_speed", "length")
    rows = []
    for m in metrics_of(records, "kernel_profile"):
        if m["label"].startswith("decode"):
            batch = int(m["label"].rsplit(" ", 1)[1])
            step_ms = decode.get(batch, {}).get("step_ms_median")
        else:
            length = int(m["label"].split()[1])
            step_ms = prefill.get(length, {}).get("ms_median")
        top = sorted(m["by_category"].items(), key=lambda kv: -kv[1]["ms"])[:3]
        rows.append(
            [
                m["label"],
                f"{m['kernels']:,}",
                f"{m['kernel_ms']:.2f}",
                f"{step_ms:.2f}" if step_ms else "—",
                f"{100 * m['kernel_ms'] / step_ms:.0f}%" if step_ms else "—",
                f"{1e3 * step_ms / m['kernels']:.1f}" if step_ms else "—",
                ", ".join(f"{name} {v['ms']:.1f} ms ({v['count']})" for name, v in top),
            ]
        )
    headers = [
        "Step",
        "GPU kernels",
        "GPU busy (ms)",
        "Step time (ms)",
        "GPU busy",
        "Step time per kernel (µs)",
        "Top kernel types",
    ]
    return markdown_table(headers, rows)


def decode_flops_bytes(cfg: ModelConfig, batch: int, context: float) -> tuple[float, float]:
    """FLOPs and bytes of one decode step (BF16), for the roofline plot."""
    embed = cfg.vocab_size * cfg.hidden_size
    matmul_params = cfg.num_params() - (0 if cfg.tie_word_embeddings else embed)  # LM head is a matmul
    attention = 4 * cfg.num_heads * cfg.head_dim * context * cfg.num_layers  # QKᵀ and PV per token
    flops = batch * (2 * matmul_params + attention)
    weight_bytes = 2 * (cfg.num_params() if cfg.tie_word_embeddings else cfg.num_params() - embed)
    return flops, weight_bytes + batch * context * cfg.kv_bytes_per_token()


def prefill_flops_bytes(cfg: ModelConfig, length: int) -> tuple[float, float]:
    """FLOPs and bytes of a prefill pass (reference attention computes the full length² score matrix)."""
    embed = cfg.vocab_size * cfg.hidden_size
    layer_params = cfg.num_params() - embed - (0 if cfg.tie_word_embeddings else embed)
    attention = 4 * cfg.num_heads * cfg.head_dim * length * length * cfg.num_layers
    flops = 2 * layer_params * length + 2 * embed + attention  # LM head for the last token only
    return flops, 2 * cfg.num_params() + length * cfg.kv_bytes_per_token()


def component_summary(records: Records) -> dict[str, dict[str, float]]:
    return {m["label"]: m["components_ms"] for m in metrics_of(records, "component_times")}


def batching_summary(records: Records) -> dict[str, Any] | None:
    return single(records, "continuous_batching")
