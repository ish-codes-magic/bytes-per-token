"""Hugging Face model cards for M4's checkpoints, generated from the raw results like every other document.

Each card says how the checkpoint was made, and shows its quality and speed next to the BF16 original's,
measured on the same GPU with the same vLLM. No number is typed by hand.
"""

from __future__ import annotations

from typing import Any

from fastserve.report.m4 import (
    DASH,
    FORMAT_LABELS,
    _f,
    _newest,
    decode_tok_s,
    kl,
    saturated,
    tasks,
    tpot_b1,
    ttft,
)
from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]
GITHUB = "https://github.com/ish-codes-magic/bytes-per-token"
SOURCES = {"c4": "C4 (English web text, allenai/c4)"}

# The published repo name per format (after "<owner>/<model>-"), and what each recipe did
REPO_SUFFIX = {"fp8": "FP8-Dynamic", "int8": "W8A8-INT8", "gptq": "W4A16-GPTQ", "awq": "W4A16-AWQ"}
TAGS = {
    "fp8": ["fp8", "w8a8"],
    "int8": ["int8", "w8a8", "smoothquant", "gptq"],
    "gptq": ["int4", "w4a16", "gptq"],
    "awq": ["int4", "w4a16", "awq"],
}
RECIPES = {
    "fp8": (
        "FP8 (E4M3) weights with one scale per output channel, and FP8 activations quantized **per token at "
        "runtime** (llm-compressor's `FP8_DYNAMIC` scheme). No calibration data is needed."
    ),
    "int8": (
        "SmoothQuant (strength 0.8) moves activation outliers into the weights, then GPTQ rounds the "
        "weights to INT8 with one scale per output channel. Activations are quantized to INT8 **per token "
        "at runtime** (llm-compressor's `W8A8` scheme)."
    ),
    "gptq": (
        "GPTQ rounds the weights to INT4 in groups of 128 (one BF16 scale per group, symmetric), "
        "compensating each rounding error with the not-yet-rounded weights. Activations stay BF16 (`W4A16`)."
    ),
    "awq": (
        "AWQ rescales the channels that carry large activations before rounding, then the weights are "
        "rounded to INT4 in groups of 128 (one BF16 scale per group, symmetric). Activations stay BF16 "
        "(`W4A16`)."
    ),
}


def repo_name(model: str, fmt: str) -> str:
    """e.g. "Qwen3-0.6B-W4A16-AWQ" (the owner is added at upload time)."""
    return f"{model.split('/')[-1]}-{REPO_SUFFIX[fmt]}"


def _one(records: Records, experiment: str, model: str, fmt: str) -> dict[str, Any] | None:
    found = [m for m in _newest(records, experiment) if m["model"] == model and m["format"] == fmt]
    return found[-1] if found else None


def _pair(rows: list[tuple[str, Any, Any, int, str]]) -> list[list[str]]:
    return [
        [label, _f(base, digits, unit), _f(this, digits, unit)] for label, base, this, digits, unit in rows
    ]


def model_card(m4: Records, m2_quality: Records, model: str, fmt: str, *, commit: str) -> str:
    """The README.md of one published checkpoint."""
    short = model.split("/")[-1]
    base_tasks, this_tasks = tasks(m4, m2_quality, model, "bf16"), tasks(m4, m2_quality, model, fmt)
    base_ppl, this_ppl = (_one(m4, "m4_vllm_perplexity", model, f) for f in ("bf16", fmt))
    base_gsm, this_gsm = (_one(m4, "m4_gsm8k", model, f) for f in ("bf16", fmt))
    calibration = checkpoint_config(m4).get("calibration")

    def strict(m: dict[str, Any] | None) -> float | None:
        return 100 * m["strict_accuracy"] if m else None

    quality = _pair(
        [
            ("KL divergence from BF16, WikiText-2 (lower is better)", 0.0, kl(m4, model, fmt), 3, ""),
            (
                "Perplexity, WikiText-2, computed by vLLM",
                (base_ppl or {}).get("perplexity"),
                (this_ppl or {}).get("perplexity"),
                2,
                "",
            ),
            ("GSM8K 5-shot, flexible-extract", base_tasks.get("gsm8k"), this_tasks.get("gsm8k"), 1, "%"),
            ("GSM8K 5-shot, strict-match", strict(base_gsm), strict(this_gsm), 1, "%"),
            (
                "MMLU 5-shot (20 questions per subject)",
                base_tasks.get("mmlu"),
                this_tasks.get("mmlu"),
                1,
                "%",
            ),
            ("HumanEval pass@1", base_tasks.get("humaneval"), this_tasks.get("humaneval"), 1, "%"),
        ]
    )
    sat = [saturated(m4, model, f) for f in ("bf16", fmt)]
    speed = _pair(
        [
            (
                "Time per output token, one user (ms)",
                tpot_b1(m4, model, "bf16"),
                tpot_b1(m4, model, fmt),
                2,
                "",
            ),
            (
                "Decode throughput, batch 1 (tokens/s)",
                decode_tok_s(m4, model, "bf16", 1),
                decode_tok_s(m4, model, fmt, 1),
                0,
                "",
            ),
            (
                "Decode throughput, batch 256 (tokens/s)",
                decode_tok_s(m4, model, "bf16", 256),
                decode_tok_s(m4, model, fmt, 256),
                0,
                "",
            ),
            (
                "Saturated server, 512 users (tokens/s)",
                *(s["output_tok_s"] if s else None for s in sat),
                0,
                "",
            ),
            (
                "Time to first token, 8k-token prompt (ms)",
                ttft(m4, model, "bf16", "long_8k"),
                ttft(m4, model, fmt, "long_8k"),
                0,
                "",
            ),
        ]
    )
    caveats = [
        "Task scores use base-model-style few-shot prompts without the chat template, identically for every "
        "format. They are for comparing formats of the same model, not for leaderboards.",
        "Speed was measured on one NVIDIA L4 (24 GB) with prefix caching off; other GPUs will differ.",
    ]
    if this_gsm and this_gsm.get("talks_past_answer", 0) > 0.1:
        caveats.append(
            f"This checkpoint often keeps writing after its final GSM8K answer "
            f"({100 * this_gsm['talks_past_answer']:.0f}% of problems, vs "
            f"{100 * (base_gsm or {}).get('talks_past_answer', 0):.0f}% for BF16). Metrics that read "
            "the *last* number in an answer (flexible-extract) under-count it; strict-match reads the "
            "marked answer."
        )
    tags = ["compressed-tensors", "llm-compressor", *TAGS[fmt], "bytes-per-token"]
    calib_line = DASH
    if fmt == "fp8":
        calib_line = "None."
    elif calibration:
        source = SOURCES.get(calibration["source"], calibration["source"])
        calib_line = f"{calibration['samples']} sequences × {calibration['seq_len']:,} tokens of {source}."
    lines = [
        "---",
        "license: apache-2.0",
        f"base_model: {model}",
        "base_model_relation: quantized",
        "library_name: vllm",
        "tags:",
        *(f"- {tag}" for tag in tags),
        "---",
        "",
        f"# {short}, {FORMAT_LABELS[fmt]}",
        "",
        f"A quantized [{model}](https://huggingface.co/{model}), made with llm-compressor and served by "
        f"vLLM. It is part of [bytes-per-token]({GITHUB}), a project that measures where every speed and "
        "cost gain in LLM serving comes from.",
        "",
        "## Use",
        "",
        "```bash",
        "vllm serve <this repo>",
        "```",
        "",
        "## How it was made",
        "",
        f"- **Method:** {RECIPES[fmt]}",
        f"- **Calibration data:** {calib_line}",
        "- **Left in BF16:** the embedding and the LM head (tied in Qwen3), and all norms.",
        "- `recipe.yaml` in this repo is llm-compressor's own record of the recipe.",
        "",
        "## Quality, against the BF16 original",
        "",
        markdown_table(["Metric", "BF16", "This checkpoint"], quality),
        "",
        "## Speed on one NVIDIA L4, vLLM 0.30.0",
        "",
        markdown_table(["Measure", "BF16", "This checkpoint"], speed),
        "",
        "## Caveats",
        "",
        *(f"- {c}" for c in caveats),
        "",
        f"Every number here is generated from `results/raw/m4_production.jsonl` in "
        f"[bytes-per-token]({GITHUB}) at commit `{commit[:10]}`; the methodology is in its M4 learning doc.",
        "",
    ]
    return "\n".join(lines)


def checkpoint_config(m4: Records) -> dict[str, Any]:
    """The settings recorded with the newest checkpoint (its calibration set), or {}."""
    found = [r for r in sorted(m4, key=lambda r: r["timestamp"]) if r["experiment"] == "m4_checkpoint"]
    return found[-1].get("config") or {} if found else {}
