"""M4 figures from a hand-made campaign."""

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from fastserve.engine.config import ModelConfig  # noqa: E402
from fastserve.viz.m4_figures import make_all  # noqa: E402


def test_m4_figures_are_written_with_one_line_captions(m4_records, tmp_path):
    models = Path(__file__).parents[2] / "benchmarks" / "models"
    configs = {
        f"Qwen/{name}": ModelConfig.from_hf(
            json.loads((models / f"{name}.config.json").read_text(encoding="utf-8"))
        )
        for name in ("Qwen3-0.6B", "Qwen3-1.7B")
    }
    hw = {"bandwidth": 262e9, "bf16": 57e12, "fp8": 118e12}
    written = make_all(m4_records, configs, hw, tmp_path)
    captions = {p.name: p.read_text(encoding="utf-8") for p in written if p.suffix == ".txt"}
    assert len(captions) == 4 and all(c.count("\n") == 1 for c in captions.values())
    assert "FP8 overtakes it from batch 256" in captions["m4_speedup_vs_batch.caption.txt"]
    assert "Qwen3-0.6B: FP8 W8A8 -17%" in captions["m4_waterfall.caption.txt"]  # 1 / 1.2 − 1
