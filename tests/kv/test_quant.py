import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.kv_cache import ContiguousKVCache  # noqa: E402
from fastserve.kv.quant import KVPolicy, apply_kv_policy, store  # noqa: E402
from fastserve.kv.sizing import KVSpec, TensorQuant  # noqa: E402


def keys_with_an_outlier_channel(seed=0, tokens=64, dim=16):
    """Keys like a real model's: every token is large in channel 3 (RoPE pairs make such channels common)."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, 2, tokens, dim, generator=g)
    k[..., 3] += 20.0
    return k


def test_bf16_and_fp8_storage():
    k = keys_with_an_outlier_channel()
    assert torch.equal(store(k, None), k)
    fp8 = store(k, TensorQuant(8, kind="fp8"))
    assert torch.equal(fp8, k.to(torch.float8_e4m3fn).float())
    assert torch.equal(
        store(torch.tensor([[[[1000.0]]]]), TensorQuant(8, kind="fp8")), torch.tensor([[[[448.0]]]])
    )


def test_per_channel_keys_isolate_the_outlier_channel():
    k = keys_with_an_outlier_channel()
    per_token = store(k, TensorQuant(4, axis="token", group=16))
    per_channel = store(k, TensorQuant(4, axis="channel", group=16))
    # Per token, the outlier stretches every token's grid, so the ordinary channels lose their precision
    normal = [c for c in range(16) if c != 3]
    token_err = (per_token - k)[..., normal].pow(2).mean()
    channel_err = (per_channel - k)[..., normal].pow(2).mean()
    assert channel_err < token_err / 10


def test_per_channel_keeps_the_residual_tokens_exact():
    k = keys_with_an_outlier_channel(tokens=70)
    out = store(k, TensorQuant(4, axis="channel", group=32))
    assert torch.equal(out[:, :, 64:], k[:, :, 64:])  # 70 mod 32 = 6 tokens wait for their group
    assert not torch.equal(out[:, :, :64], k[:, :, :64])
    assert torch.equal(store(k[:, :, :5], TensorQuant(4, axis="channel", group=32)), k[:, :, :5])


def test_rotation_helps_per_token_keys_and_is_exact_otherwise():
    k = keys_with_an_outlier_channel()
    plain = KVPolicy(KVSpec("int4", keys=TensorQuant(4, axis="token", group=16)), 16, "cpu")
    rotated = KVPolicy(
        KVSpec("int4 rot", keys=TensorQuant(4, axis="token", group=16), rotate_keys=True), 16, "cpu"
    )
    v = torch.zeros_like(k)
    err_plain = (plain.store(k, v)[0] - k).pow(2).mean()
    err_rotated = (rotated.store(k, v)[0] - k).pow(2).mean()
    assert err_rotated < err_plain / 2
    nothing = KVPolicy(KVSpec("rot only", rotate_keys=True), 16, "cpu")  # no rounding: nothing to rotate for
    assert torch.equal(nothing.store(k, v)[0], k)


def test_eviction_mask_keeps_sinks_and_the_recent_window():
    policy = KVPolicy(KVSpec("streaming", sinks=2, window=3), 16, "cpu")
    positions = torch.tensor([[9]])
    mask = torch.ones(1, 1, 1, 10, dtype=torch.bool)
    visible = policy.visible(mask, positions)[0, 0, 0].tolist()
    assert visible == [True, True, False, False, False, False, False, True, True, True]  # 0, 1 and 7, 8, 9


def test_eviction_changes_only_positions_past_the_window(tiny_model):
    ids = torch.randint(0, 256, (1, 40), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = tiny_model(ids)
        apply_kv_policy(tiny_model, KVSpec("all kept", sinks=4, window=40))
        assert torch.equal(tiny_model(ids), ref)
        apply_kv_policy(tiny_model, KVSpec("streaming", sinks=4, window=12))
        out = tiny_model(ids)
    apply_kv_policy(tiny_model, None)
    assert torch.equal(out[:, :16], ref[:, :16])  # these queries still see every earlier token
    assert not torch.allclose(out[:, 16:], ref[:, 16:])


@pytest.mark.parametrize(
    "spec",
    [
        KVSpec("int4 token", keys=TensorQuant(4, group=16), values=TensorQuant(4, group=16)),
        KVSpec("int4 kivi", keys=TensorQuant(4, axis="channel", group=8), values=TensorQuant(4, group=16)),
    ],
)
def test_quantized_cache_matches_a_full_forward_when_chunks_align(tiny_model, spec):
    """Prefilling in group-aligned chunks through a real cache stores exactly what one full pass computes."""
    ids = torch.randint(0, 256, (1, 32), generator=torch.Generator().manual_seed(1))
    apply_kv_policy(tiny_model, spec)
    with torch.no_grad():
        full = tiny_model(ids)
        cache = ContiguousKVCache(
            tiny_model.config, max_batch=1, max_len=32, dtype=torch.float32, device="cpu"
        )
        chunks = [tiny_model(ids[:, s : s + 8], torch.arange(s, s + 8)[None], cache) for s in range(0, 32, 8)]
    apply_kv_policy(tiny_model, None)
    assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-5)
    with torch.no_grad():
        assert not torch.allclose(tiny_model(ids), full, atol=1e-3)  # and the policy did change the model
