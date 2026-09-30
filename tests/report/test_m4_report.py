"""M4 report: observables for the predictions, and the speed, decode and quality tables."""

import pytest

from fastserve.report.m4 import (
    checkpoint_table,
    decode_table,
    m4_observables,
    quality_table,
    speed_table,
)
from fastserve.results import make_record


@pytest.fixture
def m2_quality():
    return [
        make_record(
            "quality_tasks", {"model": model, "scores": {"gsm8k": {"score": 0.42}}}, run_id="m2", env={}
        )
        for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
    ]


@pytest.fixture
def m3():
    metrics = {"model": "Qwen/Qwen3-0.6B", "config": "w8a8-fp8-token", "mean_kl": 0.025}
    return [make_record("m3_config", metrics, run_id="m3", env={})]


def test_observables(m4_records, m2_quality, m3):
    obs = m4_observables(m4_records, m2_quality, m3)
    assert obs["b1_fp8_small"] == pytest.approx(1.5) and obs["b1_int4_large"] == pytest.approx(2.0)
    assert obs["int4_over_fp8_b1"] == pytest.approx(2.0 / 1.5)
    assert obs["int4_over_fp8_b256"] == pytest.approx(0.9 / 1.2)
    assert obs["sat_fp8_small"] == pytest.approx(1.2) and obs["sat_int4_small"] == pytest.approx(0.9)
    assert obs["ttft8k_fp8_large"] == pytest.approx(1 / 1.5) and obs["ttft8k_int4_large"] == pytest.approx(
        1.0
    )
    assert obs["kv_fp8_small"] == pytest.approx(112_000 / 108_000)
    assert obs["gsm8k_fp8_small"] == pytest.approx(-2.0) and obs["gsm8k_int4_large"] == pytest.approx(-7.0)
    assert obs["kl_fp8_library_over_ours"] == pytest.approx(0.8) and obs["kl_int8_library"] == 0.02


def test_tables(m4_records, m2_quality):
    speed = speed_table(m4_records)
    assert speed.count("\n") == 11  # header, separator, 2 models × 5 formats
    assert "| Qwen3-0.6B | FP8 W8A8 | Fp8 |" in speed and "| 1.50× |" in speed
    decode = decode_table(m4_records, "Qwen/Qwen3-0.6B")
    assert decode.splitlines()[2].startswith("| 1 | 100 | 150 (1.50×) |")
    quality = quality_table(m4_records, m2_quality)
    assert (
        "| Qwen3-0.6B | BF16 | 0.000 | 42.0 |" in quality
        and "| INT4 W4A16 (GPTQ) | 0.300 | 35.0 |" in quality
    )
    assert checkpoint_table(m4_records).count("| 16 |") == 8


def test_fidelity_compares_served_and_simulated_perplexity(m4_records):
    from fastserve.report.m4 import fidelity_table

    table = fidelity_table(m4_records)
    assert "| Qwen3-0.6B | FP8 W8A8 | 20.40 | 20.60 | 1.0% |" in table
    assert "| Qwen3-0.6B | BF16 | 20.00 | — | — |" in table  # no vLLM BF16 perplexity in this fixture


def test_gsm8k_failures_show_bucket_shares_and_the_suite_score(m4_records, m2_quality):
    from fastserve.report.m4 import gsm8k_example, gsm8k_failure_table

    table = gsm8k_failure_table(m4_records, m2_quality)
    assert "| Qwen3-0.6B | BF16 | 40.0% | 50.0% | 0.0% | 10.0% | 120 | 42.0% |" in table  # BF16 score from M2
    assert "| Qwen3-0.6B | INT4 W4A16 (GPTQ) | 35.0% | 50.0% | 5.0% | 10.0% | 150 | 35.0% |" in table
    assert "and again and again" in gsm8k_example(m4_records, "Qwen/Qwen3-0.6B", "gptq", "looping")
    assert "No 'looping' answer" in gsm8k_example(m4_records, "Qwen/Qwen3-0.6B", "bf16", "looping")
