import json
from pathlib import Path

import pytest

from fastserve.engine.config import ModelConfig
from fastserve.report.m1 import decode_flops_bytes, m1_observables, parity_table, profile_table, speed_table

QWEN3 = Path(__file__).parents[2] / "benchmarks" / "models" / "Qwen3-0.6B.config.json"


def test_m1_observables(m1_records):
    obs = m1_observables(m1_records)
    assert obs["parity_top1"] == 100  # only the "eager" comparison counts; sdpa is the negative control
    assert obs["parity_max_abs"] == 0.0
    assert obs["decode_kernels"] == 2000
    assert obs["decode_b1_tok_s"] == pytest.approx(20.0)
    assert obs["decode_b1_gpu_busy_pct"] == pytest.approx(17.0)  # 8.5 ms busy in a 50 ms step
    assert obs["decode_b16_over_b1"] == pytest.approx(16 * 50 / 52)
    assert obs["prefill_2048_ms"] == 360.0


def test_m1_tables(m1_records):
    parity = parity_table(m1_records)
    assert parity.count("| eager |") == 2 and parity.count("| sdpa |") == 2
    assert "def f(x):…" in parity  # multi-line prompts show their first line
    assert "| decode, batch 16 | 52.0 | 308 | 15.4× |" in speed_table(m1_records)
    assert "| decode, batch 1 | 2,000 | 8.50 | 50.00 | 17% |" in profile_table(m1_records)


def test_decode_at_batch_1_has_intensity_about_1():
    cfg = ModelConfig.from_hf(json.loads(QWEN3.read_text(encoding="utf-8")))
    flops, nbytes = decode_flops_bytes(cfg, batch=1, context=128)
    assert flops / nbytes == pytest.approx(1.0, rel=0.05)
    # At batch 64 the KV cache (64 × 128 tokens × 112 KiB ≈ 0.94 GB) is almost as big as the weights
    # (1.19 GB), so intensity falls well below the "≈ batch size" rule: KV traffic catches up with weights.
    flops64, nbytes64 = decode_flops_bytes(cfg, batch=64, context=128)
    weights, kv = 2 * cfg.num_params(), 64 * 128 * cfg.kv_bytes_per_token()
    attention = 4 * cfg.num_heads * cfg.head_dim * 128 * cfg.num_layers  # QKᵀ and PV, per token
    assert nbytes64 == weights + kv
    assert flops64 == 64 * (2 * cfg.num_params() + attention)
    assert 35 < flops64 / nbytes64 < 38
