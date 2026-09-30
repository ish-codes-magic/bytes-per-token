"""Round-to-nearest integer quantization: known values, error bounds, granularity and storage cost."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.quant.rtn import IntSpec, fake_quantize, quantize  # noqa: E402


def test_known_values_int4_symmetric():
    # the worked example in PROJECT.md §6.2: max|x| = 0.9, so s = 0.9 / 7
    x = torch.tensor([[0.9, -0.2, 0.05, -0.7]])
    w_hat = fake_quantize(x, IntSpec(bits=4, granularity="tensor"))
    s = 0.9 / 7
    assert w_hat.tolist()[0] == pytest.approx([7 * s, -2 * s, 0.0, -5 * s], abs=1e-6)


def test_full_range_uses_all_16_codes_with_a_finer_step():
    x = torch.tensor([[0.75, -0.75, 0.1, 0.0]])
    full = fake_quantize(x, IntSpec(4, "tensor", full_range=True))
    s = 0.75 / 7.5  # compressed-tensors' symmetric scale
    # ±7.5 both round half-to-even to ±8: the negative end fits (−8), the positive end is clipped to 7
    assert full.tolist()[0] == pytest.approx([7 * s, -8 * s, 1 * s, 0.0])
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(3))
    spec = IntSpec(4, "group", 32)
    finer = IntSpec(4, "group", 32, full_range=True)
    assert (fake_quantize(w, finer) - w).pow(2).mean() < (fake_quantize(w, spec) - w).pow(2).mean()
    assert finer.label == "INT4 g32 sym, full range"


def test_symmetric_error_is_at_most_half_a_step():
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(0))
    spec = IntSpec(bits=4, granularity="group", group_size=32)
    _, scale, _ = quantize(w, spec)
    err = (fake_quantize(w, spec) - w).reshape(64, 8, 32).abs()
    assert (err <= scale / 2 + 1e-6).all()


def test_asymmetric_keeps_zero_exact_and_uses_at_most_2_to_the_b_levels():
    w = torch.rand(8, 64, generator=torch.Generator().manual_seed(1)) * 3 - 0.5  # mostly positive
    w[0, 0] = 0.0
    spec = IntSpec(bits=3, granularity="group", group_size=16, symmetric=False)
    w_hat = fake_quantize(w, spec)
    assert w_hat[0, 0] == 0.0
    for group in w_hat.reshape(-1, 16):
        assert len(set(group.tolist())) <= 2**3
    _, scale, _ = quantize(w, spec)
    assert ((w_hat - w).reshape(8, 4, 16).abs() <= scale + 1e-6).all()  # rounding + a rounded zero-point


def test_finer_granularity_and_more_bits_give_smaller_errors():
    g = torch.Generator().manual_seed(2)
    w = torch.randn(128, 512, generator=g) * 0.02
    w[:, 7] *= 30  # an outlier input channel, as in real weights

    def mse(spec: IntSpec) -> float:
        return (fake_quantize(w, spec) - w).pow(2).mean().item()

    tensor, channel, group = (IntSpec(4, g_) for g_ in ("tensor", "channel", "group"))
    assert mse(group) < mse(channel) < mse(tensor)
    errors = [mse(IntSpec(bits, "group")) for bits in (8, 4, 3, 2)]
    assert errors == sorted(errors)


def test_shape_dtype_and_an_all_zero_group():
    w = torch.zeros(4, 256, dtype=torch.bfloat16)
    w[1] = torch.linspace(-1, 1, 256)
    w_hat = fake_quantize(w, IntSpec(4))
    assert w_hat.shape == w.shape and w_hat.dtype == torch.bfloat16
    assert torch.isfinite(w_hat.float()).all() and (w_hat[0] == 0).all()


def test_storage_cost_counts_scales_and_zero_points():
    shape = (1024, 1024)
    assert IntSpec(4, "group", 128).bits_per_weight(shape) == 4 + 16 / 128
    assert IntSpec(4, "group", 128, symmetric=False).bits_per_weight(shape) == 4 + 20 / 128
    assert IntSpec(8, "channel").bits_per_weight(shape) == 8 + 16 / 1024
    assert IntSpec(4, "group", 128).label == "INT4 g128 sym"


def test_group_size_must_divide_the_input_features():
    with pytest.raises(ValueError):
        fake_quantize(torch.randn(4, 100), IntSpec(4, "group", 64))
