"""The quantized cache's bookkeeping (tail, group boundaries, merge), on a CPU with the reference decode.

With 16 "bits" nothing is rounded, so every difference from a plain forward pass would be a bookkeeping bug.
"""

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.kv_cache import ContiguousKVCache  # noqa: E402
from fastserve.integration.kv_cache import QuantizedKVCache  # noqa: E402
from fastserve.kv.quant import apply_kv_policy  # noqa: E402
from fastserve.kv.sizing import KVSpec, TensorQuant, bytes_per_token  # noqa: E402

SEQ, GROUP = 22, 4


def make_cache(model, bits, batch=2, max_len=SEQ, group=GROUP):
    return QuantizedKVCache(
        model.config,
        max_batch=batch,
        max_len=max_len,
        bits=bits,
        group=group,
        dtype=torch.float32,
        device="cpu",
    )


def run(model, cache, ids, prefix):
    """Prefill `prefix` tokens in one pass (none if 0), then decode one at a time: [B, SEQ, vocab]."""
    b, seq = ids.shape
    logits = []
    with torch.no_grad():
        if prefix:
            logits.append(model(ids[:, :prefix], torch.arange(prefix).expand(b, -1), cache))
        for t in range(prefix, seq):
            logits.append(model(ids[:, t : t + 1], torch.full((b, 1), t), cache))
    return torch.cat(logits, dim=1)


@pytest.fixture
def ids():
    return torch.randint(0, 256, (2, SEQ), generator=torch.Generator().manual_seed(3))


@pytest.mark.parametrize("prefix", [0, 5, 8, 21])  # no prefill; mid-group; on a group boundary; almost all
def test_unquantized_cache_equals_a_full_forward_pass(tiny_model, ids, prefix):
    with torch.no_grad():
        full = tiny_model(ids)
    got = run(tiny_model, make_cache(tiny_model, 16), ids, prefix)
    torch.testing.assert_close(got, full, atol=1e-4, rtol=1e-4)


def test_two_prefill_chunks_join_through_the_tail(tiny_model, ids):
    cache = make_cache(tiny_model, 16)
    with torch.no_grad():
        full = tiny_model(ids)
        first = tiny_model(ids[:, :6], torch.arange(6).expand(2, -1), cache)  # 4 stored, 2 waiting
        second = tiny_model(ids[:, 6:15], torch.arange(6, 15).expand(2, -1), cache)  # 12 stored, 3 waiting
    torch.testing.assert_close(torch.cat([first, second], dim=1), full[:, :15], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_prefill_matches_m5s_simulated_policy(tiny_model, ids, bits):
    """After a prefill of whole groups, the real cache holds what M5's simulation held."""
    spec = KVSpec(
        "kivi", TensorQuant(bits, axis="channel", group=GROUP), TensorQuant(bits, axis="token", group=16)
    )
    simulated = ContiguousKVCache(
        tiny_model.config, max_batch=2, max_len=SEQ, dtype=torch.float32, device="cpu"
    )
    positions = torch.arange(8).expand(2, -1)
    with torch.no_grad():
        apply_kv_policy(tiny_model, spec)
        expected = tiny_model(ids[:, :8], positions, simulated)
        apply_kv_policy(tiny_model, None)
        exact = tiny_model(ids[:, :8])
        got = tiny_model(ids[:, :8], positions, make_cache(tiny_model, bits))
    real, sim = (got - exact).abs().mean(), (expected - exact).abs().mean()
    assert 0.5 < real / sim < 2  # the same amount of rounding
    if bits == 4:
        # ...and nearly the same rounding. The real cache keeps its scales in float16, which flips the few
        # codes that sat within 2⁻¹¹ of a boundary. At 8 bits a step is so small that many do.
        assert (got - expected).abs().mean() < real / 2


def test_decoding_through_codes_stays_close_and_8_bits_is_closer(tiny_model, ids):
    with torch.no_grad():
        full = tiny_model(ids)
    error = {
        bits: (run(tiny_model, make_cache(tiny_model, bits), ids, prefix=5) - full).abs().mean().item()
        for bits in (4, 8)
    }
    assert 0 < error[8] < error[4] / 4
    assert error[4] < 0.3 * full.abs().mean().item()


def test_cache_is_append_only_and_lockstep(tiny_model, ids):
    cache = make_cache(tiny_model, 4)
    with torch.no_grad():
        tiny_model(ids[:, :5], torch.arange(5).expand(2, -1), cache)
        with pytest.raises(ValueError, match="append-only"):
            tiny_model(ids[:, 3:4], torch.full((2, 1), 3), cache)  # rewinding
        with pytest.raises(ValueError, match="append-only"):
            tiny_model(ids[:1, 5:6], torch.full((1, 1), 5), cache)  # one row of two
        with pytest.raises(ValueError, match="exceeds"):
            tiny_model(ids, torch.arange(5, 5 + SEQ).expand(2, -1), cache)
    with pytest.raises(ValueError, match="bits"):
        make_cache(tiny_model, 5)


def test_allocated_bytes_match_the_sizing_model(tiny_model):
    cfg = tiny_model.config  # 2 layers, 2 KV heads, head_dim 16
    for bits in (4, 8):
        spec = KVSpec(
            "kivi", TensorQuant(bits, axis="channel", group=32), TensorQuant(bits, axis="token", group=16)
        )
        cache = make_cache(tiny_model, bits, max_len=64, group=32)
        assert cache.bytes_per_token() == bytes_per_token(cfg, spec)
    assert make_cache(tiny_model, 16, max_len=64, group=32).bytes_per_token() == 2 * 2 * 2 * 16 * 4  # float32
