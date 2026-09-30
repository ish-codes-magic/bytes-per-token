"""The M3 driver end to end on a tiny random Qwen3 (CPU): every method builds, scores, and sizes itself."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.config import ModelConfig  # noqa: E402
from fastserve.experiments.m3 import (  # noqa: E402
    Bench,
    gptq_trace,
    histograms,
    outlier_atlas,
    run_entry,
    sensitivity,
    size_of,
    worked_example,
)

G = 16  # the tiny model's widths (64, 128) are multiples of 16, not of 128


@pytest.fixture
def bench(tiny_model):
    calib = {"source": "c4", "samples": 8, "seq_len": 32}
    windows = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(1))
    b = Bench("tiny", tiny_model, None, windows, calib)
    for samples in (4, 8, 16):  # pre-filled: no tokenizer or dataset needed
        ids = torch.randint(0, 256, (samples, 32), generator=torch.Generator().manual_seed(samples))
        b._cache[("c4", samples, 32)] = ids
    return b


ENTRIES = [
    {"name": "bf16", "method": "bf16"},
    {"name": "rtn", "method": "rtn", "bits": 4, "group_size": G},
    {"name": "nf4", "method": "nf4", "block_size": G},
    {"name": "fp8", "method": "fp8_weight", "granularity": "channel"},
    {"name": "gptq", "method": "gptq", "bits": 4, "group_size": G, "full_range": True, "block_size": G},
    {"name": "awq", "method": "awq", "bits": 4, "group_size": G, "clip": True},
    {"name": "w8a8", "method": "w8a8", "format": "fp8", "act": "token"},
    {"name": "sq", "method": "w8a8", "format": "int8", "act": "tensor", "static": True, "smooth": 0.5},
    {"name": "rot", "method": "rtn", "bits": 4, "group_size": G, "rotate": True},
]


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["name"] for e in ENTRIES])
def test_every_method_builds_and_scores(bench, entry):
    before = [p.clone() for p in bench.ref.parameters()]
    result = run_entry(bench, entry)
    assert result["config"] == entry["name"] and result["quantize_s"] >= 0
    assert result["mean_kl"] >= 0 and 0 <= result["top1_agreement"] <= 1
    if entry["method"] == "bf16":
        assert result["mean_kl"] == 0 and result["top1_agreement"] == 1
    else:
        assert result["mean_kl"] > 0
    assert all(
        torch.equal(a, b) for a, b in zip(before, bench.ref.parameters(), strict=True)
    )  # ref untouched


def test_sizes_count_scales_and_the_bf16_embedding():
    path = Path(__file__).parents[2] / "benchmarks" / "models" / "Qwen3-0.6B.config.json"
    cfg = ModelConfig.from_hf(json.loads(path.read_text(encoding="utf-8")))
    bf16 = size_of(cfg, {"method": "bf16"})
    int4 = size_of(cfg, {"method": "rtn", "bits": 4})
    rotated = size_of(cfg, {"method": "rtn", "bits": 4, "rotate": True})
    assert bf16["model_gb"] == pytest.approx(2 * cfg.num_params() / 1e9, rel=1e-3)
    assert int4["bits_per_weight"] == pytest.approx(4.125)
    assert 0.4 < int4["model_gb"] < 0.6  # far from 4× smaller: the BF16 embedding is a quarter of the weights
    assert rotated["model_gb"] - int4["model_gb"] == pytest.approx(2 * cfg.vocab_size * cfg.hidden_size / 1e9)


def test_worked_example_shows_gptq_beating_rtn():
    ex = worked_example()
    assert len(ex["steps"]) == 4 and [s["column"] for s in ex["steps"]] == [0, 1, 2, 3]
    assert ex["rtn"] == pytest.approx([0.4, 0.4, -0.4, 1.2])  # both 0.55s rounded down: errors add up
    assert ex["gptq"] == pytest.approx([0.4, 0.8, -0.4, 1.2])  # the second pushed up: errors cancel
    assert ex["loss_gptq"] < ex["loss_rtn"] / 5
    assert ex["steps"][-1]["weights"] == pytest.approx(ex["gptq"])


def test_analyses_run_and_report_what_the_figures_need(bench):
    atlas = outlier_atlas(bench, samples=4)
    assert len(atlas["residual"]) == 3 and len(atlas["residual"][0]) == 64  # 2 layer inputs + final norm
    assert len(atlas["down_input"][0]) == 128 and atlas["residual_ratio"] >= 1
    hist = histograms(bench, layer=1, group_size=G)
    assert sum(hist["weights"]) == 2 * (64 * 64 + 2 * 32 * 64 + 64 * 64 + 3 * 64 * 128) // 2
    assert "NF4" in hist["grids"] and len(hist["grids"]["INT4 (full range)"]) == 16
    trace = gptq_trace(bench, {"layer": 0, "module": "q_proj", "rows": 4, "cols": 8, "bits": 3})
    assert len(trace["frames"]) == 8 and trace["loss_gptq"] <= trace["loss_rtn"]
    scan = sensitivity(bench, {"bits": 4, "group_size": G, "windows": 1})
    assert len(scan["cells"]) == 2 * 7 and all(c["kl"] >= 0 for c in scan["cells"])


def test_folding_alone_and_float32_copies_preserve_the_model(bench):
    folded = run_entry(bench, {"name": "fold", "method": "bf16", "fold": True})
    assert folded["mean_kl"] < 1e-9 and folded["gamma_spread"]["max_over_median"] >= 1
    rotated = run_entry(bench, {"name": "rot32", "method": "bf16", "rotate": True, "dtype": "float32"})
    assert rotated["mean_kl"] < 1e-6  # float32 rounding only


def test_library_checkpoints_load_from_a_path_with_activation_quantization(bench, tmp_path):
    from safetensors.torch import save_file

    from fastserve.quant.model import decoder_linears
    from fastserve.quant.w8a8 import QuantLinear

    path = tmp_path / "tiny.dense.safetensors"
    save_file({k: v.contiguous() for k, v in bench.ref.state_dict().items()}, str(path))
    entry = {"name": "lib", "method": "library", "checkpoint": str(path), "bits": 8, "granularity": "channel"}
    plain = run_entry(bench, entry)
    w8a8 = run_entry(bench, {**entry, "act": "token", "format": "fp8"})
    assert 0 <= plain["mean_kl"] < w8a8["mean_kl"]  # BF16 storage alone, then FP8 activations on top

    from fastserve.experiments.m3 import build

    model, _ = build(bench, {**entry, "act": "token", "format": "int8"})
    assert all(isinstance(m, QuantLinear) for m in decoder_linears(model).values())
