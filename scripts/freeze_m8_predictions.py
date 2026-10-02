"""Calibrate the serving model on M2–M6 and write its predictions for every server of the M8 plan.

    uv run --only-group local python scripts/freeze_m8_predictions.py

It reads only results that existed before M8 (hw_probe, m2_serving, m4_production, m5_kv, m6_spec), so it
gives the same file whenever it is run. The file is committed before the first M8 server starts: what M8
measures is then a test of the model, not something it was fitted to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fastserve.engine.config import ModelConfig  # noqa: E402
from fastserve.report import m8  # noqa: E402
from fastserve.results import latest_run, read_jsonl  # noqa: E402

CONFIG = REPO / "benchmarks" / "configs" / "m8_ablation.yaml"
OUT = REPO / "benchmarks" / "predictions" / "m8_model.json"


def inputs() -> dict:
    """Everything the model is built from: the plan, the model configs, M0's ceilings, M2–M6's records."""
    raw = REPO / "results" / "raw"
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return {
        "config": config,
        "workloads": yaml.safe_load((REPO / config["workloads_file"]).read_text(encoding="utf-8")),
        "configs": {
            model: ModelConfig.from_pretrained_json(
                REPO / "benchmarks" / "models" / f"{model.split('/')[-1]}.config.json"
            )
            for model in (m8.SMALL, m8.LARGE)
        },
        "hw": m8.hardware(latest_run(read_jsonl(raw / "hw_probe.jsonl"))),
        **{name: read_jsonl(raw / f"{file}.jsonl") for name, file in m8.EARLIER.items()},
    }


def main() -> int:
    x = inputs()
    head = x["config"]["techniques"]["s"]["head"]
    points = m8.earlier_points(x["m2"], x["m4"], x["m5"], x["m6"], x["configs"], head, x["workloads"])
    cal = m8.calibrated(points, x["m5"], x["configs"], x["hw"])
    loads = m8.expected_loads(x["m5"], x["m6"], x["workloads"])
    starts = m8.startup_memory(x["m4"], x["m5"], x["m6"], x["configs"], head)
    frozen = m8.frozen(x["config"], x["configs"], x["hw"], cal, loads, starts)
    OUT.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {OUT.relative_to(REPO)}: {len(frozen['predictions'])} predictions from {len(points)} points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
