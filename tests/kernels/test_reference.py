"""The references define what the kernels compute, so they are checked against independent code:
nanoserve's RMSNorm and activation quantizer, M5's simulated KV storage, and the plain attention."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.attention import attention  # noqa: E402
from fastserve.engine.model import RMSNorm  # noqa: E402
from fastserve.kernels import reference  # noqa: E402
from fastserve.kv.quant import store  # noqa: E402
from fastserve.kv.sizing import KVSpec, TensorQuant  # noqa: E402
from fastserve.quant.w8a8 import W8A8Config, activation_quantizer  # noqa: E402


def random_kv(batch=2, heads=2, tokens=64, dim=16, seed=0):
    """Keys with an outlier channel (as Qwen3's have) and ordinary values."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, heads, tokens, dim, generator=g)
    k[..., 3] += 20.0
    return k, torch.randn(batch, heads, tokens, dim, generator=g)


# ---- kernel 1 --------------------------------------------------------------------------------------------


def test_rms_norm_int8_known_values():
    x = torch.tensor([[3.0, -4.0, 0.0, 0.0]])  # mean(x²) = 6.25, so y = x / 2.5
    codes, scale = reference.rms_norm_int8(x, torch.ones(4), eps=0.0)
    assert scale.item() == pytest.approx(1.6 / 127)  # the largest |y| is 1.6
    assert codes.tolist() == [[95, -127, 0, 0]]  # 1.2 / (1.6 / 127) = 95.25


def test_rms_norm_int8_is_the_models_norm_then_the_models_quantizer():
    torch.manual_seed(0)
    norm = RMSNorm(64, eps=1e-6)
    norm.weight.data = torch.rand(64) + 0.5
    x = torch.randn(5, 7, 64) * 3
    codes, scale = reference.rms_norm_int8(x, norm.weight, norm.eps)
    assert codes.shape == (5, 7, 64) and scale.shape == (5, 7, 1) and codes.dtype == torch.int8
    unfused = activation_quantizer(W8A8Config("int8"))(norm(x))  # norm, then quantize: two passes
    torch.testing.assert_close(reference.dequantize_int8(codes, scale), unfused, atol=1e-5, rtol=1e-5)


def test_rms_norm_int8_error_is_half_a_step_and_zero_rows_survive():
    torch.manual_seed(1)
    x, w = torch.randn(9, 100), torch.ones(100)
    x[4] = 0.0
    codes, scale = reference.rms_norm_int8(x, w, eps=1e-6)
    y = RMSNorm(100, 1e-6)(x)
    assert ((reference.dequantize_int8(codes, scale) - y).abs() <= scale / 2 + 1e-6).all()
    assert codes[4].abs().sum() == 0 and torch.isfinite(scale).all()
    assert codes.abs().amax(-1)[:4].eq(127).all()  # every token uses its whole range


# ---- kernel 2: storage -----------------------------------------------------------------------------------


def test_nibbles_pair_channel_d_with_d_plus_half():
    codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
    packed = reference.pack_nibbles(codes)
    assert packed.tolist() == [[0x31, 0x42]]  # channel 0 low | channel 2 high, channel 1 low | channel 3 high
    assert torch.equal(reference.unpack_nibbles(packed), codes)
    every = torch.arange(16, dtype=torch.uint8).repeat(3, 2)  # all 16 codes survive the round trip
    assert torch.equal(reference.unpack_nibbles(reference.pack_nibbles(every)), every)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_kv_is_within_half_a_step(bits):
    k, v = random_kv()
    kv = reference.quantize_kv(k, v, bits, group=32)
    assert kv.k_codes.dtype == torch.uint8 and kv.k_scale.dtype == torch.float16
    assert kv.k_codes.shape == (2, 2, 64, 8 if bits == 4 else 16)
    assert kv.k_scale.shape == (2, 2, 2, 16) and kv.v_scale.shape == (2, 2, 64)
    k_hat, v_hat = reference.dequantize_kv(kv)
    k_step = kv.k_scale.float().repeat_interleave(32, dim=2)  # [B, H, T, D]
    v_step = kv.v_scale.float()[..., None]
    # Half a step of rounding. A float16 scale can also be 2⁻¹¹ too small, which clips the top code by up
    # to 2^bits × 2⁻¹¹ of a step.
    bound = 0.51 + 2**bits * 2**-11
    assert ((k_hat - k).abs() <= k_step * bound).all()
    assert ((v_hat - v).abs() <= v_step * bound).all()


def test_more_bits_less_error_and_16_bits_is_exact():
    k, v = random_kv()
    errors = {}
    for bits in (4, 8, 16):
        k_hat, v_hat = reference.dequantize_kv(reference.quantize_kv(k, v, bits))
        errors[bits] = ((k_hat - k).pow(2).mean() + (v_hat - v).pow(2).mean()).item()
    assert errors[16] == 0.0 and errors[8] < errors[4] / 100


def test_storage_matches_m5s_simulation():
    """M5 measured quality on a simulated int4-kivi cache; the real codes must hold the same numbers."""
    k, v = random_kv(tokens=96)
    kv = reference.quantize_kv(k, v, 4, group=32)
    k_hat, v_hat = reference.dequantize_kv(kv)
    k_sim = store(k, TensorQuant(4, axis="channel", group=32))
    v_sim = store(v, TensorQuant(4, axis="token", group=16))
    # The only difference is the scale's float16 rounding (M5 kept it in float32). It moves a value by at
    # most 15 codes × 2⁻¹¹ of a step, unless the value sat that close to a rounding boundary: then the code
    # itself differs by one.
    k_diff = (k_hat - k_sim).abs() / kv.k_scale.float().repeat_interleave(32, dim=2)  # in steps
    v_diff = (v_hat - v_sim).abs() / kv.v_scale.float()[..., None]
    for diff in (k_diff, v_diff):
        assert diff.max() < 1.02 and (diff > 0.02).float().mean() < 0.03


def test_partial_groups_are_refused_and_partial_reads_work():
    k, v = random_kv(tokens=70)
    with pytest.raises(ValueError, match="whole number of groups"):
        reference.quantize_kv(k, v, 4, group=32)
    kv = reference.quantize_kv(k[:, :, :64], v[:, :, :64], 4, group=32)
    k_all, _ = reference.dequantize_kv(kv)
    k_some, v_some = reference.dequantize_kv(kv, tokens=40)  # 40 tokens: one whole group and part of the next
    assert k_some.shape == (2, 2, 40, 16) and torch.equal(k_some, k_all[:, :, :40])


def test_bytes_per_token_match_the_sizing_model():
    """The cache as allocated costs what M5's sizing formula says (one layer of Qwen3: 8 heads × 128)."""
    k = torch.zeros(1, 8, 64, 128)
    formats = {
        4: KVSpec("int4-kivi", TensorQuant(4, axis="channel", group=32), TensorQuant(4, group=128)),
        8: KVSpec("int8-kivi", TensorQuant(8, axis="channel", group=32), TensorQuant(8, group=128)),
    }
    for bits, spec in formats.items():
        expected = 2 * 8 * 128 * spec.bits_per_element() / 8
        assert reference.quantize_kv(k, k, bits).bytes_per_token() == expected
    assert reference.quantize_kv(k.half(), k.half(), 16).bytes_per_token() == 2 * 8 * 128 * 2
    assert reference.quantize_kv(k, k, 4).bytes_per_token() == 8 * 148  # the number in kernel 2's docstring


# ---- kernel 2: attention in parts --------------------------------------------------------------------------


def plain_attention(q, k, v, lengths, scale):
    """nanoserve's reference attention for one query per head: the independent oracle."""
    visible = torch.arange(k.shape[2])[None, :] < lengths[:, None]  # [B, T]
    return attention(q[:, :, None], k, v, visible[:, None, None], scale)[:, :, 0]


def test_partial_attention_is_attention():
    k, v = random_kv(batch=3)
    q = torch.randn(3, 4, 16, generator=torch.Generator().manual_seed(5))
    lengths = torch.tensor([64, 1, 37])
    out, lse = reference.partial_attention(q, k, v, lengths, scale=0.25)
    torch.testing.assert_close(out, plain_attention(q, k, v, lengths, 0.25), atol=1e-5, rtol=1e-5)
    assert lse.shape == (3, 4)


def test_merging_splits_is_exact_even_when_a_split_is_empty():
    k, v = random_kv(batch=3, tokens=96)
    q = torch.randn(3, 4, 16, generator=torch.Generator().manual_seed(6))
    lengths = torch.tensor([96, 40, 5])  # row 1 has nothing in the last split, row 2 only in the first
    outs, lses = [], []
    for start in (0, 32, 64):
        seen = (lengths - start).clamp(0, 32)
        out, lse = reference.partial_attention(
            q, k[:, :, start : start + 32], v[:, :, start : start + 32], seen, scale=0.25
        )
        outs.append(out)
        lses.append(lse)
    assert torch.isinf(lses[2][1]).all() and outs[2][1].abs().sum() == 0  # an empty split: no weight
    merged = reference.merge_partials(torch.stack(outs, dim=2), torch.stack(lses, dim=2))
    torch.testing.assert_close(merged, plain_attention(q, k, v, lengths, 0.25), atol=1e-5, rtol=1e-5)


def test_a_row_with_no_tokens_gets_zero_not_nan():
    k, v = random_kv()
    q = torch.randn(2, 4, 16)
    out, lse = reference.partial_attention(q, k, v, torch.tensor([0, 10]), scale=0.25)
    assert out[0].abs().sum() == 0 and torch.isinf(lse[0]).all()
    merged = reference.merge_partials(out[:, :, None], lse[:, :, None])
    assert torch.isfinite(merged).all() and merged[0].abs().sum() == 0


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_decode_attention_reads_the_cache_as_stored(bits):
    k, v = random_kv()
    q = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(7))
    lengths = torch.tensor([64, 33])
    kv = reference.quantize_kv(k, v, bits)
    out, lse = reference.decode_attention(q, kv, lengths, scale=0.25)
    assert out.shape == (2, 4, 1, 16) and lse.shape == (2, 4, 1)
    k_hat, v_hat = reference.dequantize_kv(kv)
    expected = plain_attention(q, k_hat, v_hat, lengths, 0.25)
    torch.testing.assert_close(reference.merge_partials(out, lse), expected, atol=1e-5, rtol=1e-5)
    if bits == 16:
        torch.testing.assert_close(expected, plain_attention(q, k, v, lengths, 0.25))
