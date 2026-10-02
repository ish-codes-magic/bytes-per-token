"""Cross-milestone summaries for the README and the interview notes: the numbers worth knowing by heart,
and the hardware and software everything ran on. Stdlib only."""

from __future__ import annotations

from typing import Any

from fastserve.perfmodel.serving import Hardware
from fastserve.report import m8
from fastserve.report.tables import markdown_table

Records = list[dict[str, Any]]
ENV_ROWS = [  # (label, key in a record's `env`)
    ("GPU", "gpu"),
    ("GPU driver", "driver"),
    ("CUDA runtime", "cuda_runtime"),
    ("Python", "python"),
    ("PyTorch", "torch"),
    ("Triton", "triton"),
    ("vLLM", "vllm"),
    ("FlashInfer", "flashinfer_python"),
    ("transformers", "transformers"),
    ("compressed-tensors", "compressed_tensors"),
    ("lm-eval", "lm_eval"),
]


def key_numbers(hw: Hardware, configs: dict[str, Any], records: Records, frozen_file: dict[str, Any]) -> str:
    """The handful of numbers the whole project hangs on, each with the milestone that measured it."""
    rows = [
        ["GPU memory bandwidth, read", f"{hw.bandwidth / 1e9:.0f} GB/s", "M0"],
        [
            "Peak matmul rate, BF16 · FP8",
            f"{hw.peak['bf16'] / 1e12:.0f} · {hw.peak['fp8'] / 1e12:.0f} TFLOP/s",
            "M0",
        ],
        ["Ridge point, BF16", f"{hw.peak['bf16'] / hw.bandwidth:.0f} FLOPs per byte", "M0"],
    ]
    for model in (m8.SMALL, m8.LARGE):
        cfg, name = configs[model], model.split("/")[-1]
        weights = 2 * cfg.num_params()
        one_user, busy = (m8.tok_s(records, model, m8.BASE, w) for w in ("m8_latency", "spec_mixed"))
        rows += [
            [
                f"{name}: weights in BF16",
                f"{weights / 1e9:.2f} GB ({cfg.num_params() / 1e6:,.0f} M parameters)",
                "M1",
            ],
            [f"{name}: KV cache per token, BF16", f"{cfg.kv_bytes_per_token(2) / 1024:.0f} KiB", "M1"],
            [
                f"{name}: one-user ceiling, bandwidth ÷ weight bytes",
                f"{hw.bandwidth / weights:.0f} tokens/s",
                "M1",
            ],
            [f"{name}: stock vLLM, one user · 64 users", f"{one_user:.0f} · {busy:,.0f} tokens/s", "M8"],
        ]
    for workload in ("m8_latency", "capacity"):
        top = m8.best(records, m8.LARGE, workload, allow_int4=False)
        base = m8.tok_s(records, m8.LARGE, m8.BASE, workload)
        what = m8.WORKLOAD_LABELS[workload].split(" (")[0].lower()
        rows.append(
            [f"Qwen3-1.7B: best measured stack, {what}", f"`{top[0]}`, {top[1] / base:.2f}× stock", "M8"]
        )
    full = m8.speedup(records, m8.LARGE, m8.FULL, "m8_latency")
    rows.append(["Qwen3-1.7B: the full stack, one user", f"`{m8.FULL}`, {full:.2f}× stock", "M8"])
    host = m8.host_step_ms(records)
    rows.append(["Host time per pass on piecewise CUDA graphs", f"about {host:.0f} ms", "M8"])
    errors = [abs(row["error"]) for row in m8.prediction_errors(records, frozen_file)]
    plain = [
        abs(row["error"])
        for row in m8.prediction_errors(records, frozen_file)
        if "s" not in m8.letters(row["label"])
    ]
    rows.append(
        [
            "Serving model, frozen: median error · without speculation",
            f"{100 * m8.fit.median_abs(errors):.1f}% · {100 * m8.fit.median_abs(plain):.1f}%",
            "M8",
        ]
    )
    return markdown_table(["Quantity", "Value", "Measured in"], rows)


def _newest_env(records: Records, experiments: tuple[str, ...] | None = None) -> dict[str, Any]:
    chosen = [r for r in records if experiments is None or r["experiment"] in experiments]
    return max(chosen, key=lambda r: r["timestamp"])["env"] if chosen else {}


def versions(serving_records: Records, research_records: Records) -> str:
    """What the two cloud images ran: the serving image (vLLM) and the research image (nanoserve, kernels)."""
    serving = _newest_env(serving_records, ("serving",))
    research = _newest_env(research_records)
    rows = [
        [label, str(serving.get(key) or "—"), str(research.get(key) or "—")]
        for label, key in ENV_ROWS
        if serving.get(key) or research.get(key)
    ]
    headers = ["", "Serving image (vLLM servers)", "Research image (nanoserve, quantizers, kernels)"]
    return markdown_table(headers, rows)
