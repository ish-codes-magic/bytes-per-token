"""The Triton kernels against their references. Runs only on a GPU (`modal run infra/modal_app.py::test_gpu`).

Shapes include widths and lengths that are not powers of two, single tokens, empty rows, GQA with one and two
query heads per KV head, and the longest context the model takes.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

from fastserve.integration.kv_cache import QuantizedKVCache  # noqa: E402
from fastserve.integration.norm_quant import quantize_norm_outputs  # noqa: E402
from fastserve.kernels import reference  # noqa: E402

FLOATS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def rand(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda").to(dtype)


# ---- kernel 1 --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(FLOATS))
@pytest.mark.parametrize("rows,d", [(1, 1024), (3, 2048), (257, 1000), (64, 3), (5, 1), (4096, 1024)])
def test_norm_quant_matches_reference(rows, d, dtype):
    from fastserve.kernels.norm_quant import rms_norm_int8

    for seed in range(3):
        x = rand(rows, d, seed=seed, dtype=FLOATS[dtype]) * 3
        weight = (rand(d, seed=seed + 100).abs() + 0.5).to(FLOATS[dtype])
        codes, scale = rms_norm_int8(x, weight, 1e-6)
        want_codes, want_scale = reference.rms_norm_int8(x, weight, 1e-6)
        assert codes.dtype == torch.int8 and codes.shape == (rows, d) and scale.shape == (rows, 1)
        torch.testing.assert_close(scale, want_scale, rtol=1e-5, atol=0)
        diff = (codes.int() - want_codes.int()).abs()
        # Same formula in the same precision. A value exactly between two codes may round the other way
        # when the two sides' sums differ in the last bit; nothing may be off by more than one code.
        assert diff.max() <= 1 and (diff > 0).float().mean() < 1e-3


def test_norm_quant_shapes_views_and_zero_rows():
    from fastserve.kernels.norm_quant import rms_norm_int8

    x = rand(2, 5, 64, dtype=torch.bfloat16)
    x[1, 2] = 0.0  # an all-zero token must not divide by zero
    weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    codes, scale = rms_norm_int8(x, weight, 1e-6)
    want_codes, want_scale = reference.rms_norm_int8(x, weight, 1e-6)
    assert codes.shape == (2, 5, 64) and scale.shape == (2, 5, 1)
    assert (codes.int() - want_codes.int()).abs().max() <= 1 and codes[1, 2].abs().sum() == 0
    assert torch.isfinite(scale).all()

    wide = rand(8, 128)  # every other column: a non-contiguous view
    codes, _ = rms_norm_int8(wide[:, ::2], weight.float(), 1e-6)
    want, _ = reference.rms_norm_int8(wide[:, ::2], weight.float(), 1e-6)
    assert (codes.int() - want.int()).abs().max() <= 1

    out = (torch.empty((8, 64), dtype=torch.int8, device="cuda"), torch.empty(8, device="cuda"))
    again, _ = rms_norm_int8(wide[:, :64].contiguous(), weight.float(), 1e-6, out=out, num_warps=8)
    assert again.data_ptr() == out[0].data_ptr()  # the caller's buffers were used


# ---- kernel 2 --------------------------------------------------------------------------------------------


def quantized(batch, kv_heads, tokens, d, bits, seed=0, dtype=torch.bfloat16):
    k = rand(batch, kv_heads, tokens, d, seed=seed, dtype=dtype)
    k[..., 3 % d] += 20.0  # an outlier channel, as Qwen3's keys have
    v = rand(batch, kv_heads, tokens, d, seed=seed + 1, dtype=dtype)
    return reference.quantize_kv(k, v, bits)


def check_attention(q, kv, lengths, scale=0.1, **launch):
    from fastserve.kernels.kv_attention import attend

    got = attend(q, kv, lengths, scale, **launch)
    want = reference.merge_partials(*reference.decode_attention(q, kv, lengths, scale))
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4 * want.abs().max().item())


@pytest.mark.parametrize("bits", [4, 8, 16])
@pytest.mark.parametrize(
    "batch,q_heads,kv_heads,tokens,d",
    [
        (1, 16, 8, 32, 128),  # Qwen3's heads, a single key group
        (2, 16, 8, 96, 128),
        (3, 4, 2, 160, 16),  # a tiny head
        (2, 4, 4, 64, 32),  # no GQA: one query head per KV head
        (1, 16, 8, 4096, 128),
    ],
)
def test_attention_matches_reference(batch, q_heads, kv_heads, tokens, d, bits):
    for seed in range(2):
        kv = quantized(batch, kv_heads, tokens, d, bits, seed=seed)
        q = rand(batch, q_heads, d, seed=seed + 7)
        full = torch.full((batch,), tokens, dtype=torch.int32, device="cuda")
        check_attention(q, kv, full)
        for split in (32, 64, tokens):  # one group per program … one program per KV head
            check_attention(q, kv, full, split=split)
        check_attention(q, kv, full, split=32, num_warps=8)


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_attention_respects_each_rows_length(bits):
    kv = quantized(4, 8, 160, 128, bits)
    q = rand(4, 16, 128, seed=3)
    lengths = torch.tensor([160, 1, 70, 0], dtype=torch.int32, device="cuda")  # full, one token, odd, none
    check_attention(q, kv, lengths, split=64)
    check_attention(q, kv, lengths, tokens=160)
    from fastserve.kernels.kv_attention import attend

    assert attend(q, kv, lengths, 0.1, split=32)[3].abs().sum() == 0  # no tokens: zeros, not NaN


def test_attention_other_dtypes_and_a_shorter_launch():
    kv = quantized(2, 8, 128, 128, 16, dtype=torch.float16)
    q = rand(2, 16, 128, seed=4, dtype=torch.float16)
    lengths = torch.tensor([64, 64], dtype=torch.int32, device="cuda")
    check_attention(q, kv, lengths)
    check_attention(q, kv, lengths, tokens=64)  # the caller knows only 64 tokens are in use: fewer splits


def test_attention_at_max_context():
    kv = quantized(1, 8, 32768, 128, 4)
    q = rand(1, 16, 128, seed=9)
    check_attention(q, kv, torch.tensor([32768], dtype=torch.int32, device="cuda"))
    check_attention(q, kv, torch.tensor([32001], dtype=torch.int32, device="cuda"), split=2048)


def test_default_split_gives_small_batches_parallelism():
    from fastserve.kernels.kv_attention import default_split

    assert default_split(1, 8, 32768) == 128  # 8 heads × 256 splits = 2,048 programs
    assert default_split(64, 8, 8192) == 256  # 128 would be 32,768 programs: capped near 16,384
    assert default_split(1, 8, 100) == 128  # never below 4 key groups
    assert all(default_split(b, 8, t) % 32 == 0 for b in (1, 3, 64) for t in (1, 1000, 30000))


# ---- inside nanoserve ------------------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_cache_with_the_kernel_matches_cache_with_the_reference(tiny_model, bits):
    from fastserve.kernels.kv_attention import decode_attention

    model = tiny_model.cuda()
    ids = torch.randint(0, 256, (2, 120), generator=torch.Generator().manual_seed(3)).cuda()

    def run(decode):
        kwargs = {} if decode is None else {"decode": decode}
        cache = QuantizedKVCache(
            model.config, max_batch=2, max_len=120, bits=bits, dtype=torch.float32, device="cuda", **kwargs
        )
        logits = []
        with torch.no_grad():
            logits.append(model(ids[:, :40], torch.arange(40, device="cuda").expand(2, -1), cache))
            for t in range(40, 120):  # crosses the group boundaries at 64 and 96
                logits.append(model(ids[:, t : t + 1], torch.full((2, 1), t, device="cuda"), cache))
        return torch.cat(logits, dim=1)

    torch.testing.assert_close(run(decode_attention), run(None), atol=2e-3, rtol=2e-3)


def test_fused_norm_quant_inside_the_model(tiny_model):
    model = tiny_model.cuda()
    ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(4)).cuda()
    with torch.no_grad():
        want = quantize_norm_outputs(model, fused=False)(ids)
        got = quantize_norm_outputs(model, fused=True)(ids)
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)
