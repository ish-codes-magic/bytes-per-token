import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from fastserve.engine.config import ModelConfig  # noqa: E402
from fastserve.perfmodel.serving import Hardware  # noqa: E402
from fastserve.viz.m8_figures import make_all  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LARGE, SMALL = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"
FIGURES = (
    "m8_waterfall",
    "m8_interactions",
    "m8_host_chain",
    "m8_recommendation",
    "m8_quality_cost",
    "m8_predicted",
    "m8_leave_one_out",
)
PREDICTIONS = [  # against the fixture: +10%, −20%, exact, +6%
    {"model": LARGE, "label": "wkps", "workload": "m8_latency", "tok_s": 237.6},
    {"model": LARGE, "label": "base", "workload": "m8_latency", "tok_s": 80.0},
    {"model": LARGE, "label": "w", "workload": "capacity", "tok_s": 312.5},
    {"model": LARGE, "label": "ws", "workload": "m8_latency", "tok_s": 228.96},
]


def test_every_figure_renders_with_a_data_driven_caption(m8_records, tmp_path):
    # The frozen file's constants (for the recommendation map), with hand-made predictions.
    frozen = json.loads((REPO / "benchmarks" / "predictions" / "m8_model.json").read_text(encoding="utf-8"))
    frozen["predictions"] = PREDICTIONS
    configs = {
        model: ModelConfig.from_pretrained_json(
            REPO / "benchmarks" / "models" / f"{model.split('/')[-1]}.config.json"
        )
        for model in (SMALL, LARGE)
    }
    hw = Hardware(bandwidth=250e9, peak={"bf16": 50e12, "fp8": 100e12, "int8": 100e12}, l2_bytes=48 * 2**20)
    perplexity = {LARGE: {"": 20.0, "w": 20.2, "k": 20.1, "wk": 20.4, "a": 22.0}}
    head = {"layers": 1, "draft_vocab_size": 32000}
    written = make_all(m8_records, frozen, configs, hw, 0.8, perplexity, head, {LARGE: 20.0}, tmp_path)
    names = {p.name for p in written}
    for figure in FIGURES:
        assert f"{figure}.png" in names and f"{figure}.svg" in names, figure
    captions = {p.name: p.read_text(encoding="utf-8") for p in written if p.suffix == ".txt"}

    waterfall = captions["m8_waterfall.caption.txt"]  # busy 1.62×, multi-turn 2.97×; the full stack is best
    assert "serves Qwen3-1.7B at 1.6–3.0× lower cost per token" in waterfall
    assert "(most on multi-turn, least on " in waterfall  # busy and long tie at 1.62×
    assert "on 0 of 5 model–workload pairs it is not the full stack" in waterfall

    interactions = captions["m8_interactions.caption.txt"]  # only FP8 weights × speculation is 0.90
    assert "20 of 24 pairs multiply to within 5%" in interactions
    assert "competes most is FP8 weights with speculative decoding on " in interactions  # 0.90 everywhere
    assert "(0.90); the pair" in interactions
    assert "Two runs of the stock server differ by up to 2%" in interactions

    chain = captions["m8_host_chain.caption.txt"]  # the fixture's piecewise servers take twice as long
    assert "8 of 8 servers on piecewise graphs are slower than their full-graph twin, by up to 2.0×" in chain
    assert "(`g` on Qwen3-1.7B: 20 ms per step against 10)" in chain

    cost = captions["m8_quality_cost.caption.txt"]  # wkps: 2,000 → 3,240 tokens/s, perplexity 20.0 → 20.4
    assert "`wkps`, costs 1.62× less than stock BF16 for a perplexity change of +2.0%" in cost

    predicted = captions["m8_predicted.caption.txt"]
    assert "the model's 4 predictions have a median error of 8% (75% within 15%)" in predicted
    assert "10% without speculation, 6% with it, and 10% where FP8 KV meets speculation" in predicted

    assert "Every technique still pays inside the full stack" in captions["m8_leave_one_out.caption.txt"]
    recommendation = captions["m8_recommendation.caption.txt"]
    assert recommendation.startswith("The M8-informed model's recommendation for Qwen3-1.7B")
    assert "of the 4 measured workloads" in recommendation
