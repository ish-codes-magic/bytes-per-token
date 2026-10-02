"""The serving model on a toy model whose every number can be checked by hand."""

from types import SimpleNamespace

import pytest

from fastserve.perfmodel.serving import (
    FLASHINFER,
    Calibration,
    Hardware,
    Load,
    Speculation,
    Stack,
    attention_efficiency,
    dollars_per_million,
    kv_time,
    kv_tokens_read,
    linear_params,
    predict,
    prefill_flops,
    prefill_time,
    running_batch,
    step_time,
    time_per_token,
    weight_bytes,
    weight_time,
)

# 1,000 layer parameters and a tied 10 × 10 head; 100 KV bytes per token in BF16.
CFG = SimpleNamespace(
    vocab_size=10,
    hidden_size=10,
    num_layers=2,
    num_heads=4,
    head_dim=5,
    tie_word_embeddings=True,
    num_params=lambda: 1_100,
    kv_bytes_per_token=lambda bytes_per_elem=2: 50 * bytes_per_elem,
)
HW = Hardware(
    bandwidth=1_000.0, peak={"bf16": 10_000.0, "fp8": 20_000.0, "int8": 20_000.0}
)  # bytes/s, FLOP/s
PLAIN = Calibration()


def test_weights_cost_bytes_once_and_flops_per_token():
    assert linear_params(CFG) == 1_000
    assert weight_bytes(CFG, "bf16") == 2_200 and weight_bytes(CFG, "fp8") == 1_200  # the head stays BF16
    assert weight_bytes(CFG, "int4") == pytest.approx(1_000 * 4.125 / 8 + 200)
    # one token: 2.2 s of reading + (2,000 + 200) FLOPs at 10,000 FLOP/s
    assert weight_time(CFG, HW, Stack(), 1) == pytest.approx(2.2 + 0.22)
    assert weight_time(CFG, HW, Stack(), 10) == pytest.approx(2.2 + 2.2)  # reading doesn't grow, math does
    # FP8: half the layer bytes and twice the layer FLOP/s; the head is unchanged
    assert weight_time(CFG, HW, Stack(weights="fp8"), 10) == pytest.approx(1.2 + 1.0 + 0.2)
    assert weight_time(CFG, HW, Stack(weights="int4"), 10) == pytest.approx(0.715625 + 2.2)  # BF16 math


def test_kv_term_scales_with_batch_context_and_format():
    assert kv_time(CFG, HW, Stack(), PLAIN, batch=2, context=30) == pytest.approx(2 * 30 * 100 / 1_000)
    fp8 = Stack(kv="fp8", backend=FLASHINFER)
    assert kv_time(CFG, HW, fp8, PLAIN, batch=2, context=30) == pytest.approx(3.0)
    cal = Calibration(flash_attn_batch_penalty=0.25, flashinfer_efficiency={"bf16": 1.0, "fp8": 0.5})
    assert attention_efficiency(Stack(), cal, 1) == 1.0 and attention_efficiency(Stack(), cal, 16) == 0.5
    assert kv_time(CFG, HW, fp8, cal, batch=2, context=30) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="FlashInfer"):
        Stack(kv="fp8")


def test_step_adds_overheads():
    cal = Calibration(step_s=1.0, per_seq_s=0.1, act_quant_s=0.5)
    base = weight_time(CFG, HW, Stack(), 4) + kv_time(CFG, HW, Stack(), cal, 4, 10)
    assert step_time(CFG, HW, Stack(), cal, batch=4, context=10) == pytest.approx(1.0 + 0.4 + base)
    w8a8 = Stack(weights="fp8")
    plain = weight_time(CFG, HW, w8a8, 4) + kv_time(CFG, HW, w8a8, cal, 4, 10)
    assert step_time(CFG, HW, w8a8, cal, batch=4, context=10) == pytest.approx(1.9 + plain)
    # verifying 3 tokens per sequence: three times the math, the same bytes
    extra = step_time(CFG, HW, Stack(), cal, 4, 10, new_tokens=3) - step_time(CFG, HW, Stack(), cal, 4, 10)
    assert extra == pytest.approx(2 * 4 * 2_200 / 10_000)


def test_prefill_is_linear_plus_quadratic():
    linear, attention = prefill_flops(CFG, new_tokens=10)
    assert linear == 2 * 1_000 * 10 and attention == 4 * (10 * 5) * 5 * 4 * 2  # 50 causal pairs
    _, after_cache = prefill_flops(CFG, new_tokens=10, cached=20)
    assert after_cache == 4 * (10 * 25) * 5 * 4 * 2  # each new token also attends to the 20 cached ones
    assert prefill_time(CFG, HW, Stack(), PLAIN, 10) == pytest.approx(2.0 + 0.8 + 2.2)  # + one weight read
    cal = Calibration(prefill_linear={"bf16": 0.5}, prefill_attention=0.25)
    assert prefill_time(CFG, HW, Stack(), cal, 10) == pytest.approx(4.0 + 3.2 + 2.2)
    assert prefill_time(CFG, HW, Stack(), PLAIN, 0) == 0.0  # fully cached: nothing to run


def test_one_user_is_prefill_then_steps():
    load = Load(users=1, prompt_len=10, output_len=4)
    out = predict(CFG, HW, Stack(), PLAIN, load)
    step = step_time(CFG, HW, Stack(), PLAIN, 1, context=12)
    prefill = prefill_time(CFG, HW, Stack(), PLAIN, 10)
    assert out["tok_s"] == pytest.approx(4 / (prefill + 4 * step))
    assert out["tpot_ms"] == pytest.approx(1e3 * step) and out["ttft_ms"] == pytest.approx(
        1e3 * (prefill + step)
    )
    assert out["batch"] == 1 and 0 < out["kv_share"] < 1


def test_a_batch_shares_each_step_and_prefix_caching_shrinks_prefill():
    load = Load(users=8, prompt_len=10, output_len=4, cached_len=6)
    off = predict(CFG, HW, Stack(), PLAIN, load)
    step = step_time(CFG, HW, Stack(), PLAIN, 8, context=12)
    # With others decoding, a prompt rides in their next pass: it adds math, not another read of the weights
    prefill = prefill_time(CFG, HW, Stack(), PLAIN, 10, own_pass=False)
    assert prefill == pytest.approx(2.0 + 0.8)
    assert off["tok_s"] == pytest.approx(4 / (prefill + 4 * step / 8))
    assert off["tpot_ms"] == pytest.approx(1e3 * (step + prefill * 7 / 4))  # others' prefills cut in
    on = predict(CFG, HW, Stack(prefix_caching=True), PLAIN, load)
    cached = prefill_time(CFG, HW, Stack(), PLAIN, 4, cached=6, own_pass=False)
    assert on["prefill_ms"] == pytest.approx(1e3 * cached)
    assert on["tok_s"] > off["tok_s"] and on["step_ms"] == off["step_ms"]


def test_one_user_also_waits_for_request_handling():
    cal = Calibration(request_s=0.5)
    load = Load(users=1, prompt_len=10, output_len=4)
    plain, slow = predict(CFG, HW, Stack(), PLAIN, load), predict(CFG, HW, Stack(), cal, load)
    assert slow["ttft_ms"] == pytest.approx(plain["ttft_ms"] + 500)
    assert 4 / slow["tok_s"] == pytest.approx(4 / plain["tok_s"] + 0.5)  # the GPU idles meanwhile
    busy = Load(users=8, prompt_len=10, output_len=4)
    assert predict(CFG, HW, Stack(), cal, busy)["tok_s"] == predict(CFG, HW, Stack(), PLAIN, busy)["tok_s"]


def test_a_shared_prefix_is_read_once_per_group_if_a_layer_of_it_fits_in_l2():
    """8 sequences, 30 tokens of context each, 20 of them a prefix shared by groups of 4."""
    load = Load(users=8, prompt_len=28, output_len=4, shared_len=20, sharers=4)
    on = Stack(prefix_caching=True)
    separate = kv_tokens_read(CFG, HW, Stack(), 8, 30, load)
    assert separate == 8 * 30  # prefix caching off: every sequence has its own copy
    roomy = Hardware(bandwidth=1_000.0, peak=HW.peak, l2_bytes=1_000)  # a layer's slice: 20 × 100 / 2 bytes
    assert kv_tokens_read(CFG, roomy, on, 8, 30, load) == 8 * 10 + 2 * 20  # two groups read the prefix
    tight = Hardware(bandwidth=1_000.0, peak=HW.peak, l2_bytes=999)
    assert kv_tokens_read(CFG, tight, on, 8, 30, load) == 8 * 30  # it does not fit: no sharing
    assert kv_time(CFG, roomy, on, PLAIN, 8, 30, load) == pytest.approx((80 + 40) * 100 / 1_000)
    alone = Load(users=8, prompt_len=28, output_len=4, shared_len=20, sharers=1)
    assert kv_tokens_read(CFG, roomy, on, 8, 30, alone) == 8 * 30


def test_cache_capacity_limits_the_batch():
    cal = Calibration(kv_slack=1.25)
    load = Load(users=96, prompt_len=30, output_len=10, kv_tokens=1_000)
    assert running_batch(load, cal) == pytest.approx(1_000 / (40 * 1.25))  # 20 sequences fit
    assert running_batch(Load(users=4, prompt_len=30, output_len=10, kv_tokens=1_000), cal) == 4
    assert running_batch(Load(users=500, prompt_len=1, output_len=1), cal) == 256  # vLLM's own limit
    bigger = Load(users=96, prompt_len=30, output_len=10, kv_tokens=2_000)
    assert predict(CFG, HW, Stack(), cal, bigger)["batch"] == pytest.approx(40)


def test_speculation_pays_only_if_tokens_per_pass_beat_its_cost():
    spec = Speculation(k=3, tokens_per_pass=2.0, draft_bytes=100.0)
    cal = Calibration(draft_step_s=0.05, draft_per_seq_s=0.01)
    plain = time_per_token(CFG, HW, Stack(), cal, batch=2, context=10)
    verify = step_time(CFG, HW, Stack(), cal, 2, 10, new_tokens=4)
    drafted = 3 * (0.05 + 100 / 1_000 + 0.01 * 2)
    got = time_per_token(CFG, HW, Stack(speculation=spec), cal, batch=2, context=10)
    assert got == pytest.approx((verify + drafted) / 2.0)
    assert got < plain  # two tokens per pass for a pass that costs less than two steps
    useless = Speculation(k=3, tokens_per_pass=1.0, draft_bytes=100.0)  # nothing accepted: all cost
    assert time_per_token(CFG, HW, Stack(speculation=useless), cal, 2, 10) > plain


def test_on_piecewise_graphs_a_pass_waits_for_the_slower_of_gpu_and_host():
    spec = Speculation(k=3, tokens_per_pass=2.0, draft_bytes=100.0)
    assert not Stack(speculation=spec).piecewise  # FlashAttention keeps the full graph with speculation
    assert not Stack(backend=FLASHINFER).piecewise  # and FlashInfer keeps it without
    collide = Stack(kv="fp8", backend=FLASHINFER, speculation=spec)
    assert collide.piecewise
    gpu_pass = 2.0 * time_per_token(CFG, HW, collide, PLAIN, batch=1, context=10)  # no host time modeled
    slow_host = Calibration(host_step_s=10 * gpu_pass)
    assert time_per_token(CFG, HW, collide, slow_host, 1, 10) == pytest.approx(10 * gpu_pass / 2.0)
    fast_host = Calibration(host_step_s=gpu_pass / 10)  # the GPU is the slower one: nothing changes
    assert time_per_token(CFG, HW, collide, fast_host, 1, 10) == pytest.approx(gpu_pass / 2.0)
    # a stack that keeps its full graph never waits for the host
    full = Stack(speculation=spec)
    assert time_per_token(CFG, HW, full, slow_host, 1, 10) == time_per_token(CFG, HW, full, PLAIN, 1, 10)


def test_cost():
    assert dollars_per_million(1_000, dollars_per_hour=0.8) == pytest.approx(0.8 / 3.6)
