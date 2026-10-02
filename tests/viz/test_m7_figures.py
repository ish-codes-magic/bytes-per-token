from types import SimpleNamespace

import pytest

pytest.importorskip("matplotlib")

from fastserve.viz.m7_figures import make_all  # noqa: E402

FIGURES = ("m7_roofline", "m7_speedup", "m7_tuning", "m7_timeline", "m7_traffic", "m7_waterfall")
QWEN3_SMALL = SimpleNamespace(hidden_size=1024, head_dim=128)


def test_every_figure_renders_with_a_data_driven_caption(m7_records, tmp_path):
    written = make_all(m7_records, 250e9, QWEN3_SMALL, 0.8, tmp_path)
    names = {p.name for p in written}
    for figure in FIGURES:
        assert f"{figure}.png" in names and f"{figure}.svg" in names, figure
    captions = {p.name: p.read_text(encoding="utf-8") for p in written if p.suffix == ".txt"}

    roofline = captions["m7_roofline.caption.txt"]  # 134 / 111 / 38 GB/s of 250; kernel 1: 252 of 250
    assert "BF16 cache at 54% of the measured bandwidth and the INT4 cache at 44%" in roofline
    assert "PyTorch's path manages 15%" in roofline and "at 101% once" in roofline

    speedup = captions["m7_speedup.caption.txt"]  # 1 × 512 is the fixture's launch-bound loss: 0.35 / 0.5
    assert "0.7–10× the speed of nanoserve's PyTorch attention (1 of 3 shapes slower)" in speedup
    assert "0.14–2.00× FlashInfer's" in speedup

    tuning = captions["m7_tuning.caption.txt"]
    assert "best cell is 512 tokens per program with 4 warps (512 programs)" in tuning
    assert "not splitting is 10× slower, and the wrong warp count at that split (1) 1.4×" in tuning

    timeline = captions["m7_timeline.caption.txt"]  # 10 + 40 + 10 busy of 80 before; 10 + 5 + 10 of 45 after
    assert "works 60 of 80 ms before and 25 of 45 ms after" in timeline and "3 small kernels" in timeline

    traffic = captions["m7_traffic.caption.txt"]  # d = 1,024: 7d + 4 → 3d + 4; INT4: 148 + 512 + 512 → 148
    assert "2.3× fewer bytes per token in kernel 1 (7,172 → 3,076)" in traffic
    assert "7.9× fewer per cached token in kernel 2 (1,172 → 148)" in traffic
    assert "a BF16 cache costs 512 bytes" in traffic

    waterfall = captions["m7_waterfall.caption.txt"]  # 140 → 70 ms and 140 → 56 ms per token
    assert "nanoserve at 1 × 32,000 tokens" in waterfall
    assert "kernel 2 changes cost by -50%, and INT4 codes by -60% (KL 0.02" in waterfall


def test_figures_missing_their_data_are_skipped_not_fatal(m7_records, tmp_path):
    profile_only = [r for r in m7_records if r["experiment"] == "m7_profile"]
    written = make_all(profile_only, 250e9, QWEN3_SMALL, 0.8, tmp_path)
    assert {p.name for p in written} == {"m7_traffic.png", "m7_traffic.svg", "m7_traffic.caption.txt"}
