"""M3 figures, drawn from a synthetic campaign shaped like the real records."""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("plotly")

from fastserve.results import make_record  # noqa: E402
from fastserve.viz.m3_figures import FIGURES, make_all  # noqa: E402


@pytest.fixture
def campaign(m3_records):
    import random

    rng = random.Random(0)
    extra = []

    def add(experiment, metrics):
        extra.append(
            make_record(experiment, metrics, run_id="m3test", env={}, git={}, config={"task": "analyses"})
        )

    edges = [i / 100 - 1 for i in range(201)]
    add(
        "m3_histograms",
        {
            "model": "Qwen/Qwen3-0.6B",
            "layer": 14,
            "edges": edges,
            "weights": [int(1000 * (1 - abs(e))) for e in edges[:-1]],
            "activations": [1000 if abs(e) < 0.05 else 3 for e in edges[:-1]],
            "activation_abs_max_over_median": 250.0,
            "grids": {
                "INT4 (textbook)": [i / 7 for i in range(-7, 8)],
                "INT4 (full range)": [i / 7.5 for i in range(-8, 8)],
                "NF4": [
                    -1,
                    -0.7,
                    -0.5,
                    -0.4,
                    -0.28,
                    -0.18,
                    -0.09,
                    0,
                    0.08,
                    0.16,
                    0.25,
                    0.34,
                    0.44,
                    0.56,
                    0.72,
                    1,
                ],
                "FP8 E4M3": [-1, -0.5, -0.25, 0, 0.25, 0.5, 1],
            },
        },
    )
    residual = [[rng.uniform(0.5, 2) for _ in range(64)] for _ in range(29)]
    for row in residual[3:]:
        row[7] = 900.0  # an outlier channel from layer 3 on
    add(
        "m3_outliers",
        {
            "model": "Qwen/Qwen3-0.6B",
            "residual": residual,
            "down_input": [[rng.uniform(0.1, 5) for _ in range(128)] for _ in range(28)],
            "residual_rotated": [[rng.uniform(5, 30) for _ in range(64)] for _ in range(29)],
            "residual_ratio": 700.0,
            "residual_rotated_ratio": 3.2,
            "down_input_ratio": 40.0,
            "top_channels": [7, 1, 2, 3, 4],
        },
    )
    w = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(4)]
    frames = [{"column": i, "weights": [[v + 0.01 * i for v in row] for row in w]} for i in range(8)]
    add(
        "m3_gptq_trace",
        {
            "model": "Qwen/Qwen3-0.6B",
            "layer": 1,
            "module": "q_proj",
            "spec": "INT3 per-channel sym",
            "weights": w,
            "rtn": w,
            "frames": frames,
            "loss_rtn": 2.0,
            "loss_gptq": 0.5,
        },
    )
    return m3_records + extra


def test_m3_figures_are_written_with_one_line_captions(campaign, tmp_path):
    written = make_all(campaign, tmp_path)
    for name in FIGURES:
        assert tmp_path / f"{name}.png" in written
        caption = (tmp_path / f"{name}.caption.txt").read_text(encoding="utf-8")
        assert caption.count("\n") == 1
    assert tmp_path / "m3_gptq_motion.html" in written
    assert "700×" in (tmp_path / "m3_outlier_atlas.caption.txt").read_text(encoding="utf-8")
    assert "4.0× lower" in (tmp_path / "m3_gptq_motion.caption.txt").read_text(encoding="utf-8")
