"""Calibration recovers the constants that generated a set of points, and uses each point where it says."""

from types import SimpleNamespace

import pytest

from fastserve.perfmodel import fit
from fastserve.perfmodel.fit import Point
from fastserve.perfmodel.serving import FLASHINFER, Calibration, Hardware, Load, Speculation, Stack, predict

MODEL = "toy"
CFG = SimpleNamespace(
    vocab_size=1_000,
    hidden_size=100,
    num_layers=4,
    num_heads=4,
    head_dim=25,
    tie_word_embeddings=True,
    num_params=lambda: 2_100_000,
    kv_bytes_per_token=lambda bytes_per_elem=2: 500 * bytes_per_elem,
)
CONFIGS = {MODEL: CFG}
HW = Hardware(bandwidth=1e9, peak={"bf16": 1e11, "fp8": 2e11, "int8": 2e11})
TRUE = Calibration(
    step_s=1e-3,
    per_seq_s=2e-5,
    act_quant_s=4e-4,
    flash_attn_batch_penalty=0.05,
    flashinfer_efficiency={"bf16": 0.9, "fp8": 0.8},
    prefill_linear={"bf16": 0.7, "fp8": 0.5},
    prefill_attention=0.6,
    draft_step_s=3e-4,
    draft_per_seq_s=1e-5,
    kv_slack=1.1,
    request_s=5e-3,
)


def point(group, label, stack, users, prompt, output, workload="w", kv_tokens=None) -> Point:
    """A point whose "measurement" is exactly what the true constants predict."""
    load = Load(users=users, prompt_len=prompt, output_len=output, kv_tokens=kv_tokens)
    out = predict(CFG, HW, stack, TRUE, load)
    ttft = out["ttft_ms"] if users == 1 else None
    return Point("m0", group, MODEL, label, workload, users, stack, load, out["tok_s"], out["tpot_ms"], ttft)


def campaign() -> list[Point]:
    plain, fp8 = Stack(), Stack(weights="fp8")
    eagle = Stack(speculation=Speculation(k=3, tokens_per_pass=2.0, draft_bytes=1e5))
    points = [point("long", "bf16", plain, 1, n, 16) for n in (2_000, 8_000, 32_000)]
    points += [point("long", "fp8", fp8, 1, n, 16) for n in (2_000, 8_000)]
    points += [point("decode", "bf16", plain, users, 100, 200) for users in (1, 4, 16, 64, 256)]
    points += [point("decode", "fp8", fp8, users, 100, 200) for users in (1, 4, 16)]
    points += [point("capacity", "bf16", plain, 96, 4_000, 200, kv_tokens=200_000)]
    points += [point("saturation", "bf16", plain, 256, 400, 300)]
    points += [point("capacity", "fi", Stack(backend=FLASHINFER), 96, 4_000, 200, kv_tokens=200_000)]
    points += [point("long", "fi", Stack(backend=FLASHINFER), 1, 32_000, 16)]
    fp8_kv = Stack(kv="fp8", backend=FLASHINFER)
    points += [point("capacity", "fp8kv", fp8_kv, 96, 4_000, 200, kv_tokens=400_000)]
    points += [point("long", "fp8kv", fp8_kv, 1, 32_000, 16)]
    points += [point("spec", "bf16-eagle3-k3", eagle, users, 100, 200) for users in (1, 64)]
    points += [point("spec", "held-out", eagle, 16, 100, 200), point("prefix", "held-out", plain, 8, 500, 50)]
    return points


def test_golden_section_and_coordinate_descent():
    assert fit.golden_section(lambda x: (x - 0.3) ** 2, 0, 1) == pytest.approx(0.3, abs=1e-6)
    best = fit.minimize(lambda v: (v["a"] - 2) ** 2 + (v["b"] + 1) ** 2, {"a": (0, 5), "b": (-3, 3)})
    assert best["a"] == pytest.approx(2, abs=1e-4) and best["b"] == pytest.approx(-1, abs=1e-4)


def test_each_stage_uses_only_its_own_points():
    points = campaign()
    use = fit.stages(points)
    assert {p.label for p in use["overhead"]} == {"bf16"} and {p.users for p in use["overhead"]} == {1, 4, 16}
    assert {p.users for p in use["flash_attn"]} == {64, 96, 256}
    assert {p.label for p in use["act_quant"]} == {"fp8"}
    assert [p.load.prompt_len for p in use["prefill"]] == [2_000, 8_000, 32_000]
    assert {p.label for p in use["flashinfer_bf16"]} == {"fi"} and {
        p.label for p in use["flashinfer_fp8"]
    } == {"fp8kv"}
    assert [p.users for p in use["draft_step"]] == [1] and [p.users for p in use["draft_per_seq"]] == [64]
    held_out = [p for p in points if id(p) not in fit.fitted_on(points)]
    assert {p.label for p in held_out} == {"held-out"}


def test_calibration_recovers_the_constants_that_made_the_points():
    points = campaign()
    cal = fit.calibrate(points, CONFIGS, HW, kv_slack=TRUE.kv_slack)
    assert cal.step_s == pytest.approx(TRUE.step_s, rel=0.05)
    assert cal.per_seq_s == pytest.approx(TRUE.per_seq_s, rel=0.2)
    assert cal.flash_attn_batch_penalty == pytest.approx(TRUE.flash_attn_batch_penalty, rel=0.2)
    assert cal.act_quant_s == pytest.approx(TRUE.act_quant_s, rel=0.1)
    assert cal.flashinfer_efficiency["bf16"] == pytest.approx(0.9, rel=0.03)
    assert cal.flashinfer_efficiency["fp8"] == pytest.approx(0.8, rel=0.03)
    assert cal.prefill_linear["bf16"] == pytest.approx(0.7, rel=0.05)
    assert cal.prefill_linear["fp8"] == pytest.approx(0.5, rel=0.05)
    assert cal.prefill_attention == pytest.approx(0.6, rel=0.05)
    assert cal.request_s == pytest.approx(TRUE.request_s, rel=0.1)
    assert cal.draft_step_s == pytest.approx(TRUE.draft_step_s, rel=0.1)
    assert cal.draft_per_seq_s == pytest.approx(TRUE.draft_per_seq_s, rel=0.15)
    # Points no stage used are predicted too: that is what "held out" buys
    errors = fit.errors(points, CONFIGS, HW, cal)
    assert len(errors) == len(points) and fit.median_abs(errors) < 0.01 and max(abs(e) for e in errors) < 0.05


def test_errors_are_signed_and_skip_missing_metrics():
    points = campaign()[:2]
    assert fit.errors(points, CONFIGS, HW, TRUE) == pytest.approx([0.0, 0.0], abs=1e-12)
    slow = Calibration(**{**TRUE.__dict__, "prefill_attention": 0.3})  # prefill twice as slow in attention
    assert all(e < 0 for e in fit.errors(points, CONFIGS, HW, slow))  # under-predicts throughput
    many_users = [p for p in campaign() if p.users > 1][:3]
    assert fit.errors(many_users, CONFIGS, HW, TRUE, "ttft_ms") == []  # TTFT is a one-user quantity here
    assert fit.median_abs([]) is None
