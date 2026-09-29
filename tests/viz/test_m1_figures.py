import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("plotly")

from fastserve.engine.config import ModelConfig  # noqa: E402
from fastserve.viz.m1_figures import make_all  # noqa: E402

QWEN3 = Path(__file__).parents[2] / "benchmarks" / "models" / "Qwen3-0.6B.config.json"
NAMES = ["m1_decode_scaling", "m1_anatomy", "m1_roofline", "m1_kv_growth", "m1_attention", "m1_block_table"]


def test_m1_figures_are_written_with_one_line_captions(m1_records, probe_records, tmp_path):
    cfg = ModelConfig.from_hf(json.loads(QWEN3.read_text(encoding="utf-8")))
    written = make_all(m1_records, probe_records, cfg, tmp_path)
    for name in NAMES:
        for suffix in (".png", ".svg", ".caption.txt"):
            assert tmp_path / f"{name}{suffix}" in written
        assert (tmp_path / f"{name}.caption.txt").read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "m1_block_table.html") in written
    assert "handed to new ones 1 times" in (tmp_path / "m1_block_table.caption.txt").read_text(
        encoding="utf-8"
    )
