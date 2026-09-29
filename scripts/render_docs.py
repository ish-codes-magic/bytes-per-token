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
from fastserve.hw.analysis import m0_observables  # noqa: E402
from fastserve.report.m1 import m1_observables, parity_table, profile_table, speed_table  # noqa: E402
from fastserve.report.render import render_file  # noqa: E402
from fastserve.report.tables import (  # noqa: E402
    decode_matmul_table,
    hw_summary,
    model_facts,
    prediction_table,
)
from fastserve.results import latest_run, read_jsonl  # noqa: E402


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
