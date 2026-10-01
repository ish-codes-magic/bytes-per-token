"""M5 report: observables for the predictions, and the tables."""

from types import SimpleNamespace

import pytest

from fastserve.report.m5 import (
    kv_serving_table,
    m5_observables,
    policy_table,
    prefix_table,
    saturation_table,
    vllm_quality_table,
)
from fastserve.results import make_record

QWEN3 = SimpleNamespace(num_layers=28, num_kv_heads=8, head_dim=128)


@pytest.fixture
def m4():
    metrics = {"model": "Qwen/Qwen3-0.6B", "format": "bf16", "perplexity": 20.0}
    return [make_record("m4_vllm_perplexity", metrics, run_id="m4", env={})]


@pytest.fixture
def m2_quality():
    metrics = {"model": "Qwen/Qwen3-0.6B", "pass_rate": 1.0}
    return [make_record("quality_needle", metrics, run_id="m2", env={})]


def test_observables(m5_records, m4):
    obs = m5_observables(m5_records, m4)
    assert obs["kv_tokens_fp8_small"] == 2.0
    assert obs["capacity_running_fp8_small"] == 2.0
    assert obs["sat_tput_fp8_small"] == pytest.approx(1.5)
    assert obs["sat_flashinfer_small"] == pytest.approx(1.1)
    assert obs["tpot32k_fp8_small"] == pytest.approx(1.5)
    assert obs["kl_kivi_over_token"] == pytest.approx(0.02)
    assert obs["kl_kivi_large_over_small"] == pytest.approx(2.0)
    assert obs["key_over_value_outliers"] == pytest.approx(4.0)  # medians 8 / 2
    assert obs["needle_streaming"] == 0.5 and obs["needle_vllm_fp8_small"] == 1.0
    assert obs["ppl_vllm_fp8_small"] == pytest.approx(1.01)
    assert obs["prefix_cached_share"] == pytest.approx(3 * 1600 / 8000)  # the first request finds nothing
    assert obs["prefix_ttft_small"] == pytest.approx(0.25)
    assert obs["prefix_tput_small"] == pytest.approx(1.5)
    assert obs["needle_int4_token"] is None  # not in the fixture: no crash, just a dash


def test_tables(m5_records, m4, m2_quality, m5_policies):
    policies = policy_table(m5_records, m5_policies, QWEN3, kv_bytes=18 * 2**30)
    assert "| FP8 E4M3, scale 1.0 | 8.00 | 56.0 |" in policies
    assert "| StreamingLLM, 1,024 kept | 16.00 | 112.0 | 164 |" in policies  # 18 GiB / (1,024 × 112 KiB)
    assert "| BF16 (control) | 16.00 | 112.0 | 5 | 0.0000 | 0.0000 | 90.0% | 100% |" in policies
    serving = kv_serving_table(m5_records)
    assert "| Qwen3-0.6B | FP8 KV | FLASHINFER | 340,000 | 80 |" in serving
    prefix = prefix_table(m5_records)
    assert "| Qwen3-0.6B | multi-turn | on | 60.0% | 50 | 10.00 | 750 |" in prefix
    quality = vllm_quality_table(m5_records, m2_quality, m4)
    assert "| Qwen3-0.6B | 20.00 | 20.20 | 1.00% | 100% | 100% |" in quality


def test_saturation_table_shows_the_step(m5_records):
    table = saturation_table(m5_records, bandwidth=2**30 * 17.2 / 0.05)  # streams 17.2 GiB in 50 ms
    # 20 steps/s → 50 ms; 1 GiB of weights + 0.9 × 18 GiB of KV = 17.2 GiB = 18.5 GB: all of the step
    assert "| Qwen3-0.6B | FP8 KV | 250 | 90% | 50 | 18.5 | 100% | 50 | 0 | 3,000 |" in table
