from types import SimpleNamespace

import pytest

pytest.importorskip("matplotlib")

from fastserve.viz.m5_figures import make_all  # noqa: E402

QWEN3 = SimpleNamespace(num_layers=28, num_kv_heads=8, head_dim=128)
WORKLOADS = {
    "multi_turn": {
        "shared_prefix_len": 20,
        "prefixes": 2,
        "turns": 2,
        "reply_len": 5,
        "input_len": 4,
        "output_len": 4,
        "num_requests": 8,
    }
}
FIGURES = (
    "m5_needle",
    "m5_key_value_channels",
    "m5_concurrency",
    "m5_radix_tree",
    "m5_prefix_ttft",
    "m5_waterfall",
)


def test_every_figure_renders_with_a_data_driven_caption(m5_records, m5_policies, tmp_path):
    written = make_all(m5_records, m5_policies, QWEN3, WORKLOADS, tmp_path)
    names = {p.name for p in written}
    for figure in FIGURES:
        assert f"{figure}.png" in names, figure
    captions = {p.name: p.read_text(encoding="utf-8") for p in written if p.suffix == ".txt"}
    assert "StreamingLLM, 1,024 kept (50%)" in captions["m5_needle.caption.txt"]
    assert "8.0× the median one, against 2.0× for values" in captions["m5_key_value_channels.caption.txt"]
    assert "ran 40 (BF16) vs 80 (FP8)" in captions["m5_concurrency.caption.txt"]
    assert "8 prompts" in captions["m5_radix_tree.caption.txt"]
    assert "TTFT p50 200 → 50 ms" in captions["m5_prefix_ttft.caption.txt"]
    assert "saturation -38%" in captions["m5_waterfall.caption.txt"]  # 1 / 1.6 − 1
