"""W8A8 simulation: which activation scales survive outliers, and the quantized linear layer itself."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.quant.w8a8 import QuantLinear, W8A8Config, activation_quantizer, quantize_weight  # noqa: E402


def relative_error(x, x_hat):
    return ((x_hat - x).pow(2).mean() / x.pow(2).mean()).item()


def activations(outlier_channel=True, outlier_token=False):
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(0))
    if outlier_channel:
        x[:, 17] *= 60  # a massive-activation channel: in every token
    if outlier_token:
        x[5] *= 60  # one extreme token
    return x


def test_an_outlier_token_ruins_per_tensor_int8_but_not_fp8():
    x = activations(outlier_channel=False, outlier_token=True)
    normal = torch.ones(64, dtype=torch.bool)
    normal[5] = False

    def error(fmt, granularity):
        x_hat = activation_quantizer(W8A8Config(fmt, act_granularity=granularity))(x)
        return relative_error(x[normal], x_hat[normal])

    # INT8's even grid: the outlier token's max sets a step far too coarse for everyone else
    assert error("int8", "token") < error("int8", "tensor") / 10
    # FP8's grid is logarithmic: relative precision is the same at any scale, so one scale is nearly enough
    assert error("fp8", "tensor") < 2 * error("fp8", "token")


def test_fp8_tolerates_an_outlier_channel_better_than_int8():
    x = activations()
    for granularity in ("tensor", "token"):
        fp8 = activation_quantizer(W8A8Config("fp8", act_granularity=granularity))(x)
        int8 = activation_quantizer(W8A8Config("int8", act_granularity=granularity))(x)
        assert relative_error(x[:, :17], fp8[:, :17]) < relative_error(x[:, :17], int8[:, :17])


def test_a_static_scale_saturates_values_beyond_its_calibrated_range():
    x = torch.tensor([[1.0, -2.0, 8.0]])
    q = activation_quantizer(W8A8Config("fp8", act_granularity="tensor", act_static=True), torch.tensor(4.0))
    assert q(x)[0, 2].item() == pytest.approx(4.0)  # clipped at the calibrated max


def test_quant_linear_computes_with_both_operands_quantized():
    linear = torch.nn.Linear(256, 32, bias=True)
    cfg = W8A8Config("int8", act_granularity="token")
    act = activation_quantizer(cfg)
    layer = QuantLinear(linear, quantize_weight(linear.weight.data, cfg), act)
    x = activations()
    expected = torch.nn.functional.linear(act(x), quantize_weight(linear.weight.data, cfg), linear.bias)
    assert torch.allclose(layer(x), expected)
    plain = activations(outlier_channel=False)
    assert relative_error(linear(plain), layer(plain)) < 1e-3  # close to the original, not equal
