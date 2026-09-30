"""Channel scaling (shared by AWQ and SmoothQuant) preserves the model; AWQ's searches do what they claim."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.quant.awq import AWQConfig, awq_group, search_clip, search_scale  # noqa: E402
from fastserve.quant.equivalence import input_groups, scale_inputs  # noqa: E402
from fastserve.quant.rtn import IntSpec, fake_quantize  # noqa: E402
from fastserve.quant.w8a8 import smooth_group  # noqa: E402

IDS = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))


def test_scaling_any_group_leaves_the_outputs_unchanged(make_tiny_qwen3):
    _, model = make_tiny_qwen3()
    before = model(IDS)
    for layer in model.model.layers:
        for group in input_groups(layer):
            n = group.linears[0].in_features
            scale_inputs(group, torch.rand(n, generator=torch.Generator().manual_seed(n)) * 3 + 0.2)
    assert torch.allclose(model(IDS), before, atol=1e-4)


def linear_problem(seed=0):
    """A linear layer, and inputs where channel 3 is 20× more active than the others."""
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(64, 48, bias=False)
    lin.weight.data = torch.randn(48, 64, generator=g)
    x = torch.randn(512, 64, generator=g)
    x[:, 3] *= 20
    return lin, x


def test_alpha_zero_without_duo_scaling_is_plain_rtn():
    lin, x = linear_problem()
    w = lin.weight.data.clone()
    cfg = AWQConfig(IntSpec(3, "group", 16), duo_scaling=False)
    _, _, losses = search_scale([lin], x.abs().mean(0), lambda: x @ lin.weight.T, cfg)
    rtn = (x @ fake_quantize(w, cfg.spec).T - x @ w.T).pow(2).mean().item()
    assert losses[0] == pytest.approx(rtn, rel=1e-5)
    assert torch.equal(lin.weight.data, w)  # the search leaves the weights as it found them


@pytest.mark.parametrize("duo", [False, True])
def test_the_search_protects_a_salient_channel(duo):
    lin, x = linear_problem()
    cfg = AWQConfig(IntSpec(3, "group", 16), duo_scaling=duo)
    s, alpha, losses = search_scale([lin], x.abs().mean(0), lambda: x @ lin.weight.T, cfg)
    assert alpha > 0 and min(losses) < 0.9 * losses[0]
    assert s[3] == s.max()  # the busiest channel gets the biggest scale


def test_weight_means_are_relative_to_each_groups_max():
    w = torch.ones(4, 32)
    w[:, 0] = 4.0  # the max of the first group of 16
    from fastserve.quant.awq import weight_means

    means = weight_means([w, w], IntSpec(4, "group", 16))
    assert means.shape == (32,) and means[0] == pytest.approx(1.0)
    assert means[1] == pytest.approx(0.25) and means[16] == pytest.approx(1.0)


def test_clipping_never_increases_the_output_error():
    lin, x = linear_problem(seed=1)
    w = lin.weight.data
    cfg = AWQConfig(IntSpec(3, "channel"))  # one group per row: each row's choice is exactly optimal
    clipped = search_clip(w, x, cfg)
    assert (clipped.abs() <= w.abs() + 1e-6).all()

    def err(weights):
        return (x @ fake_quantize(weights, cfg.spec).T - x @ w.T).pow(2).mean()

    assert err(clipped) <= err(w) + 1e-6


def test_awq_group_is_function_preserving_before_rounding(make_tiny_qwen3):
    _, model = make_tiny_qwen3()
    before = model(IDS)
    layer = model.model.layers[0]
    group = input_groups(layer)[1]  # gate/up after the post-attention norm
    x = torch.randn(2, 16, 64, generator=torch.Generator().manual_seed(4))  # the MLP's input
    mlp = layer.mlp
    s, result = awq_group(group, x.abs().mean((0, 1)), lambda: mlp(x), AWQConfig(IntSpec(4, "group", 16)))
    assert s.shape == (64,) and len(result["losses"]) == 20
    assert torch.allclose(model(IDS), before, atol=1e-4)


def test_smoothquant_shrinks_an_outlier_channel_and_preserves_the_model(make_tiny_qwen3):
    _, model = make_tiny_qwen3()
    before = model(IDS)
    group = input_groups(model.model.layers[1])[0]
    act_amax = torch.ones(64)
    act_amax[7] = 50.0  # an outlier channel
    s = smooth_group(group, act_amax, alpha=0.5)
    assert s[7] == s.max() and (act_amax / s).max() < act_amax.max() / 3
    assert torch.allclose(model(IDS), before, atol=1e-4)
