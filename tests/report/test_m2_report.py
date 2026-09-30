import pytest

from fastserve.report.m2 import (
    SMALL,
    knee_rate,
    load_table,
    long_context_table,
    m2_observables,
    summary_table,
)


def test_m2_observables(m2_records):
    obs = m2_observables(m2_records, nanoserve_b1_tok_s=20.0)
    assert obs["chat_tpot_ms"] == 6.5
    assert obs["chat_ttft_ms"] == 20
    assert obs["vllm_over_nanoserve"] == pytest.approx((1000 / 6.5) / 20)
    assert obs["peak_tok_s"] == 4000  # the closed-loop point beats every open-loop point here
    assert obs["knee_rate"] == 8  # 16 req/s only kept 50% within the SLO
    assert obs["tpot_ratio_1_7b"] == pytest.approx(2.5)
    assert obs["throughput_ratio_1_7b"] == pytest.approx(1 / 2.5)
    assert obs["long32k_ttft_s"] == pytest.approx(0.7)
    assert obs["long32k_tpot_ratio"] == pytest.approx(20 / 6.5)
    assert obs["shared_prefix_peak_rate"] == 14.0


def test_knee_is_none_when_nothing_meets_the_slo(m2_records):
    assert knee_rate(m2_records, SMALL, threshold=1.01) is None


def test_m2_tables(m2_records):
    assert "| Qwen3-0.6B | 20.0 | 6.50 | 154 | 4,000 | 8 |" in summary_table(m2_records)
    table = load_table(m2_records, SMALL)
    assert "| 16 req/s |" in table and "| 64 users |" in table and "| 50% |" in table
    assert "| Qwen3-0.6B | 32k | 700 | 20.00 |" in long_context_table(m2_records)


def test_newest_per_model_keeps_one_record_per_experiment_and_model(m2_records):
    from fastserve.report.m2 import newest_per_model

    older = dict(m2_records[0], timestamp="2000-01-01T00:00:00+00:00", metrics={**m2_records[0]["metrics"]})
    kept = newest_per_model([older, m2_records[0]])
    assert kept == [m2_records[0]]


def test_offline_table_compares_engine_with_served_peak(m2_records):
    from fastserve.report.m2 import offline_table

    offline = [
        {
            "experiment": "offline_throughput",
            "metrics": {"model": "Qwen/Qwen3-0.6B", "output_throughput": 8000.0},
        }
    ]
    assert "| Qwen3-0.6B | 8,000 | 4,000 | 50% |" in offline_table(offline, m2_records)
