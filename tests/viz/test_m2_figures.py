import pytest

pytest.importorskip("matplotlib")

from fastserve.viz.m2_figures import FIGURES, make_all  # noqa: E402


def test_m2_figures_are_written_with_one_line_captions(m2_records, tmp_path):
    written = make_all(m2_records, tmp_path)
    for name in FIGURES:
        for suffix in (".png", ".svg", ".caption.txt"):
            assert tmp_path / f"{name}{suffix}" in written
        assert (tmp_path / f"{name}.caption.txt").read_text(encoding="utf-8").count("\n") == 1
    assert "up to 8 req/s" in (tmp_path / "m2_goodput.caption.txt").read_text(encoding="utf-8")


def test_saturation_figure_caption_reads_the_plateau(m2_saturation_records, tmp_path):
    import json
    from pathlib import Path

    from fastserve.engine.config import ModelConfig
    from fastserve.viz.m2_figures import make_saturation

    path = Path(__file__).parents[2] / "benchmarks" / "models" / "Qwen3-0.6B.config.json"
    cfg = ModelConfig.from_hf(json.loads(path.read_text(encoding="utf-8")))
    written = make_saturation(m2_saturation_records, {"Qwen/Qwen3-0.6B": cfg}, 262e9, tmp_path)
    assert tmp_path / "m2_saturation.png" in written
    caption = (tmp_path / "m2_saturation.caption.txt").read_text(encoding="utf-8")
    assert "Held full for 8 s" in caption and "4,000 tokens/s with 200 sequences" in caption
