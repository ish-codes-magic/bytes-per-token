"""Fill the generated blocks in docs/ and README.md from results/raw/, so no number is typed by hand.

Stdlib only, so it runs on the laptop:

    uv run --only-group local python scripts/render_docs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # the laptop doesn't install the package, only its stdlib-only modules

from fastserve.engine.config import ModelConfig  # noqa: E402
from fastserve.hw.analysis import (  # noqa: E402
    m0_observables,
    measured_bandwidth,
    measured_peak_flops,
    metrics_of,
)
from fastserve.report import m4 as m4_report  # noqa: E402
from fastserve.report.m1 import m1_observables, parity_table, profile_table, speed_table  # noqa: E402
from fastserve.report.m2 import (  # noqa: E402
    LARGE,
    SMALL,
    decode_efficiency_table,
    load_table,
    long_context_table,
    m2_observables,
    newest_per_model,
    offline_table,
    peak_definition_table,
    prefill_budget_table,
    prefill_efficiency_table,
    quality_observables,
    quality_table,
    repeat_table,
    saturation_table,
    summary_table,
)
from fastserve.report.m3 import (  # noqa: E402
    config_table,
    m3_observables,
    model_size_table,
    names_in_task,
    sensitivity_table,
    worked_example_block,
)
from fastserve.report.render import render_file  # noqa: E402
from fastserve.report.tables import (  # noqa: E402
    decode_matmul_table,
    hw_summary,
    model_facts,
    prediction_table,
)
from fastserve.results import latest_run, read_jsonl  # noqa: E402


def m4_blocks(m4: list, m2_quality: list, m3: list) -> dict[str, str]:
    predictions = json.loads((REPO / "benchmarks" / "predictions" / "m4.json").read_text(encoding="utf-8"))
    return {
        "m4_predictions": prediction_table(predictions, m4_report.m4_observables(m4, m2_quality, m3)),
        "m4_speed": m4_report.speed_table(m4),
        "m4_decode_small": m4_report.decode_table(m4, "Qwen/Qwen3-0.6B"),
        "m4_decode_large": m4_report.decode_table(m4, "Qwen/Qwen3-1.7B"),
        "m4_quality": m4_report.quality_table(m4, m2_quality),
        "m4_checkpoints": m4_report.checkpoint_table(m4),
        "m4_fidelity": m4_report.fidelity_table(m4),
    }


def compute_spend(path: Path) -> str:
    """One line: what the cloud compute has cost so far, from Modal's billing (scripts/compute_log.py)."""
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_resource: dict[str, float] = {}
    for row in rows:
        by_resource[row["resource"]] = by_resource.get(row["resource"], 0.0) + float(row["usd"])
    ranked = sorted(by_resource.items(), key=lambda kv: -kv[1])
    parts = ", ".join(f"{name} ${usd:.2f}" for name, usd in ranked)
    through = rows[-1]["billed_through_hour"].replace("T", " ")[:16]
    total = sum(by_resource.values())
    return f"Cloud compute so far: **${total:.2f}** on Modal ({parts}), billed through {through} UTC."


def m3_blocks(m3: list) -> dict[str, str]:
    predictions = json.loads((REPO / "benchmarks" / "predictions" / "m3.json").read_text(encoding="utf-8"))
    int4 = ["rtn-int4-g128-full", "gptq-int4-g128", "gptq-int4-g128-trueseq", "library-gptq-int4-g128"]
    int4 += ["awq-int4-g128", "awq-int4-g128-clip", "awq-int4-g128-paper", "library-awq-int4-g128"]
    int4 += ["rtn-int4-channel", "gptq-int4-channel", "rtn-int3-g128", "gptq-int3-g128", "awq-int3-g128-clip"]
    rotation = ["rtn-int4-channel", "rot-rtn-int4-channel", "rtn-int4-g128", "rot-rtn-int4-g128"]
    rotation += [
        "gptq-int4-g128",
        "rot-gptq-int4-g128",
        "w8a8-int8-tensor-static",
        "rot-w8a8-int8-tensor-static",
    ]
    return {
        "m3_predictions": prediction_table(predictions, m3_observables(m3)),
        "m3_grids": config_table(m3, names_in_task(m3, "grids"), reference="rtn-int4-g128"),
        "m3_calibrated": config_table(m3, int4, reference="rtn-int4-g128-full"),
        "m3_rotation": config_table(m3, rotation),
        "m3_rotation_diagnosis": config_table(
            m3, ["rtn-int4-g128", *names_in_task(m3, "rotation_diagnosis"), "rot-rtn-int4-g128"]
        ),
        "m3_w8a8": config_table(m3, names_in_task(m3, "w8a8"), reference="w8a8-int8-tensor-static"),
        "m3_calibration": config_table(
            m3, ["gptq-int4-g128", *names_in_task(m3, "calibration")], reference="gptq-int4-g128"
        ),
        "m3_model_size": model_size_table(m3, names_in_task(m3, "large", model="Qwen/Qwen3-1.7B")),
        "m3_sensitivity": sensitivity_table(m3),
        "m3_gptq_worked_example": worked_example_block(m3),
    }


def build_blocks() -> dict[str, str]:
    blocks: dict[str, str] = {}
    probe = REPO / "results" / "raw" / "hw_probe.jsonl"
    if probe.exists():
        records = latest_run(read_jsonl(probe))
        blocks["hw_summary"] = hw_summary(records)
        blocks["m0_decode_matmuls"] = decode_matmul_table(records)
        predictions = json.loads(
            (REPO / "benchmarks" / "predictions" / "m0.json").read_text(encoding="utf-8")
        )
        blocks["m0_predictions"] = prediction_table(predictions, m0_observables(records))
        qwen3 = ModelConfig.from_pretrained_json(REPO / "benchmarks" / "models" / "Qwen3-0.6B.config.json")
        blocks["m1_model_facts"] = model_facts(qwen3, "Qwen3-0.6B", records)
    nanoserve = REPO / "results" / "raw" / "m1_nanoserve.jsonl"
    if nanoserve.exists():
        m1 = latest_run(read_jsonl(nanoserve))
        predictions = json.loads(
            (REPO / "benchmarks" / "predictions" / "m1.json").read_text(encoding="utf-8")
        )
        blocks["m1_predictions"] = prediction_table(predictions, m1_observables(m1))
        blocks["m1_parity"] = parity_table(m1)
        blocks["m1_speed"] = speed_table(m1)
        blocks["m1_profile"] = profile_table(m1)
    serving = REPO / "results" / "raw" / "m2_serving.jsonl"
    if serving.exists() and nanoserve.exists():
        m2 = latest_run(read_jsonl(serving))
        b1 = next(m["tokens_per_s"] for m in metrics_of(m1, "decode_speed") if m["batch"] == 1)
        predictions = json.loads(
            (REPO / "benchmarks" / "predictions" / "m2.json").read_text(encoding="utf-8")
        )
        blocks["m2_predictions"] = prediction_table(predictions, m2_observables(m2, nanoserve_b1_tok_s=b1))
        blocks["m2_summary"] = summary_table(m2)
        blocks["m2_load_small"] = load_table(m2, SMALL)
        blocks["m2_load_large"] = load_table(m2, LARGE)
        blocks["m2_shared_prefix"] = load_table(m2, SMALL, "shared_prefix")
        blocks["m2_long_context"] = long_context_table(m2)
        configs = {
            model: ModelConfig.from_pretrained_json(
                REPO / "benchmarks" / "models" / f"{model.split('/')[-1]}.config.json"
            )
            for model in (SMALL, LARGE)
        }
        peak = measured_peak_flops(latest_run(read_jsonl(probe)))["bf16"]
        blocks["m2_prefill"] = prefill_efficiency_table(m2, configs, peak)
        saturation = REPO / "results" / "raw" / "m2_saturation.jsonl"
        if saturation.exists():
            sat = latest_run(read_jsonl(saturation))
            bandwidth = measured_bandwidth(latest_run(read_jsonl(probe)))
            blocks["m2_peak_definition"] = peak_definition_table(m2, sat)
            blocks["m2_saturation"] = saturation_table(sat, configs, bandwidth)
            blocks["m2_decode_efficiency"] = decode_efficiency_table(m2, sat, configs, bandwidth)
            blocks["m2_prefill_budget"] = prefill_budget_table(sat)
            blocks["m2_repeat"] = repeat_table(m2, sat)
        offline = REPO / "results" / "raw" / "m2_offline.jsonl"
        if offline.exists():
            blocks["m2_offline"] = offline_table(newest_per_model(read_jsonl(offline)), m2)
    quality = REPO / "results" / "raw" / "m2_quality.jsonl"
    if quality.exists():
        m2q = newest_per_model(read_jsonl(quality))
        predictions = json.loads(
            (REPO / "benchmarks" / "predictions" / "m2_quality.json").read_text(encoding="utf-8")
        )
        blocks["m2_quality_predictions"] = prediction_table(predictions, quality_observables(m2q))
        blocks["m2_quality"] = quality_table(m2q)
    quant = REPO / "results" / "raw" / "m3_quant.jsonl"
    if quant.exists():
        blocks.update(m3_blocks(read_jsonl(quant)))  # every run: tasks can be re-run, the newest result wins
    production = REPO / "results" / "raw" / "m4_production.jsonl"
    if production.exists() and quant.exists() and quality.exists():
        blocks.update(m4_blocks(read_jsonl(production), read_jsonl(quality), read_jsonl(quant)))
    spend = REPO / "results" / "compute_log.csv"
    if spend.exists():
        blocks["compute_spend"] = compute_spend(spend)
    # Each figure's one-line takeaway, written by the figure code: <!-- BEGIN GENERATED: caption-<name> -->
    for caption in (REPO / "results" / "figures").glob("*.caption.txt"):
        name = caption.name.removesuffix(".caption.txt")
        blocks[f"caption-{name}"] = f"*{caption.read_text(encoding='utf-8').strip()}*"
    return blocks


def main() -> int:
    blocks = build_blocks()
    missing_any = False
    for path in sorted([REPO / "README.md", *REPO.glob("docs/**/*.md")]):
        changed, missing = render_file(path, blocks)
        if changed:
            print(f"updated {path.relative_to(REPO)}")
        for name in missing:
            missing_any = True
            print(f"no data yet for block '{name}' in {path.relative_to(REPO)}")
    return 1 if missing_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
