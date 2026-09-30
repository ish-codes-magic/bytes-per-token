"""Whole-model entry points on a tiny random Qwen3: each method runs, lands on its grid, behaves sensibly."""

import copy

import pytest

torch = pytest.importorskip("torch")

from fastserve.quant.awq import AWQConfig  # noqa: E402
from fastserve.quant.gptq import GPTQConfig  # noqa: E402
from fastserve.quant.model import (  # noqa: E402
    awq_model,
    decoder_linears,
    gptq_model,
    input_amax,
    quantize_weights,
    smoothquant_model,
    w8a8_model,
)
from fastserve.quant.rtn import IntSpec, fake_quantize  # noqa: E402
from fastserve.quant.w8a8 import QuantLinear, W8A8Config  # noqa: E402

CALIB = torch.randint(0, 256, (8, 32), generator=torch.Generator().manual_seed(1))
EVAL = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(2))


def kl(ref, cand):
    p, q = torch.log_softmax(ref, -1), torch.log_softmax(cand, -1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def on_grid(model, bits, group):
    for w in decoder_linears(model).values():
        for g in w.weight.reshape(-1, group):
            assert len(set(g.tolist())) <= 2**bits


def test_decoder_linears_finds_seven_per_layer(tiny_model):
    names = list(decoder_linears(tiny_model))
    assert (
        len(names) == 2 * 7
        and names[0] == "layers.0.self_attn.q_proj"
        and names[-1] == "layers.1.mlp.down_proj"
    )


def test_int8_rtn_barely_changes_the_model(tiny_model):
    before = tiny_model(EVAL)
    quantize_weights(tiny_model, lambda w: fake_quantize(w, IntSpec(8, "channel")))
    assert kl(before, tiny_model(EVAL)) < 1e-3


def test_gptq_model_lands_on_the_grid_and_lowers_each_objective(tiny_model):
    spec = IntSpec(3, "group", 16)
    stats = gptq_model(tiny_model, CALIB, GPTQConfig(spec, block_size=16), batch=4)
    assert len(stats) == 14
    assert sum(s["gptq_loss"] for s in stats) < sum(s["rtn_loss"] for s in stats)
    on_grid(tiny_model, 3, 16)


def test_awq_model_lands_on_the_grid(tiny_model):
    before = tiny_model(EVAL)
    stats = awq_model(tiny_model, CALIB, AWQConfig(IntSpec(4, "group", 16)), batch=4)
    assert len(stats) == 2 * 3 and all(0 <= s["alpha"] < 1 for s in stats)
    on_grid(tiny_model, 4, 16)
    assert torch.isfinite(tiny_model(EVAL)).all() and kl(before, tiny_model(EVAL)) < 0.5


def test_smoothquant_then_static_w8a8(tiny_model):
    before = tiny_model(EVAL)
    amax = input_amax(tiny_model, CALIB, batch=4)
    assert amax["layers.0.mlp.down_proj"].shape == (128,)
    raw = copy.deepcopy(amax)
    smoothquant_model(tiny_model, amax, alpha=0.5)
    assert not torch.equal(amax["layers.0.self_attn.q_proj"], raw["layers.0.self_attn.q_proj"])
    w8a8_model(tiny_model, W8A8Config("int8", act_granularity="tensor", act_static=True), amax)
    assert all(isinstance(m, QuantLinear) for m in decoder_linears(tiny_model).values())
    assert kl(before, tiny_model(EVAL)) < 0.05


def test_static_scales_need_calibration(tiny_model):
    with pytest.raises(ValueError):
        w8a8_model(tiny_model, W8A8Config("fp8", act_granularity="tensor", act_static=True))
