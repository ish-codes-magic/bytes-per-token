"""M7 report: observables for the predictions, and the tables."""

from types import SimpleNamespace

import pytest

from fastserve.report.m7 import (
    attention_error_table,
    attention_table,
    flush_table,
    m4_gap,
    m4_gap_table,
    m7_observables,
    nanoserve_table,
    norm_exactness_table,
    norm_in_model_table,
    norm_quant_table,
    norm_warps_table,
    ops_profile_table,
    profile_table,
    quality_table,
    quant_launch_cost,
    roofline_table,
    timing_views_table,
    tune_table,
)

BANDWIDTH = 250e9  # bytes/s


def test_observables(m7_records):
    obs = m7_observables(m7_records, BANDWIDTH)
    assert obs["nq_graph_1"] == pytest.approx(2.0)
    assert obs["nq_graph_4096"] == pytest.approx(2.0)
    assert obs["nq_graph_32768"] == pytest.approx(2.5)
    assert obs["nq_bandwidth"] == pytest.approx(32768 * 6148 / 0.8e-3 / BANDWIDTH)
    assert obs["nq_vs_fused_fp8"] == pytest.approx(2.0)
    assert obs["nq_eager_1"] == pytest.approx(2.0)  # (0.004 + 0.056) / (0.002 + 0.028)
    assert obs["nq_codes_reference"] == pytest.approx(0.9998) and obs["nq_codes_vllm"] == pytest.approx(0.96)
    assert obs["att_bf16_kernel_vs_sdpa_32k"] == pytest.approx(3.5)
    assert obs["att_int4_vs_bf16_kernel_32k"] == pytest.approx(1 / 0.35)
    assert obs["att_int8_vs_bf16_kernel_32k"] == pytest.approx(1 / 0.7)
    assert obs["att_bf16_kernel_vs_flashinfer_32k"] == pytest.approx(0.7)
    assert obs["att_int4_vs_flashinfer_32k"] == pytest.approx(2.0)
    assert obs["att_int4_vs_dequant_32k"] == pytest.approx(100.0)
    assert obs["att_int4_bandwidth_32k"] == pytest.approx(8 * 32768 * 148 / 0.35e-3 / BANDWIDTH)
    assert obs["att_int4_vs_sdpa_512"] == pytest.approx(0.35 / 0.5)  # launch-bound: slower than PyTorch
    assert obs["att_int4_error"] == pytest.approx(2e-5)
    assert obs["tune_best_split_32k"] == 512 and obs["tune_nosplit_penalty"] == pytest.approx(10.0)
    assert obs["ns_int4_512"] == pytest.approx(0.9) and obs["ns_int4_32k"] == pytest.approx(2.5)
    assert obs["ns_int4_b8"] == pytest.approx(2.0) and obs["ns_dequant_32k"] == pytest.approx(0.2)
    assert obs["ns_memory"] == pytest.approx(33152 / 114688)
    assert obs["kl_int4"] == pytest.approx(0.02) and obs["needle_int4"] == 1.0


def test_observables_survive_a_campaign_that_only_profiled(m7_records):
    profile_only = [r for r in m7_records if r["config"]["task"] in ("", "ops")]
    assert all(value is None for value in m7_observables(profile_only, BANDWIDTH).values())


def test_profile_tables(m7_records):
    profile = profile_table(m7_records, BANDWIDTH)
    # 3.2 GB of KV in 100 ms of attention: 32 GB/s, 13% of the bandwidth
    assert "| 1 × 32,000 | 140.0 | 100.0 | 71% | 2,000 | 3.20 | 32 | 13% |" in profile
    ops = ops_profile_table(m7_records)
    assert "| 1 × 2,048 | 2.0 | 2.0 | 4.0 | 60.0 | 3.0 |" in ops
    assert "Kernel 1" not in ops  # the "before" table never shows the kernel


def test_kernel_1_tables(m7_records):
    table = norm_quant_table(m7_records, "graph")
    assert "| 32,768 × 2,048 | 30.0 | 2,000.0 | 1,600.0 | 800.0 | 2.50× | 2.00× | 252 |" in table
    assert "| 1 × 2,048 | 300.0 | 60.0 | 50.0 | 30.0 | 2.00× |" in norm_quant_table(m7_records, "eager")
    exact = norm_exactness_table(m7_records)
    assert "| 1 × 2,048 | 100.000% | 1 | 96.00% | 1 |" in exact and "| 99.980% |" in exact
    assert "| 256 × 2,048 | 900.0 | 300.0 ← | 600.0 |" in norm_warps_table(m7_records)


def test_m4_gap_is_split_into_launches_and_the_rest(m7_records, m4_records):
    cfg = SimpleNamespace(
        vocab_size=10,
        hidden_size=2048,
        num_layers=10,
        tie_word_embeddings=True,
        num_params=lambda: 1_100,
        kv_bytes_per_token=lambda: 1.0,
    )
    cost = quant_launch_cost(m7_records, cfg)
    assert cost == {"launches": 40, "us_each": pytest.approx(2.0), "ms": pytest.approx(0.08)}
    gap = m4_gap(m4_records, cfg, 1e9, "Qwen/Qwen3-0.6B")
    assert gap["gap_ms"] == pytest.approx(gap["measured_ms"] - gap["predicted_ms"])
    table = m4_gap_table(m7_records, m4_records, {"Qwen/Qwen3-0.6B": cfg}, 1e9)
    assert "| Qwen3-0.6B |" in table and "40 × 2.00 µs" in table


def test_kernel_2_tables(m7_records):
    table = attention_table(m7_records)
    assert "| 1 × 32,768 | 3,500 | 700 | 35,000 | 1,000 | 700 | 350 | 10.00× | 2.00× |" in table
    assert "| 16 × 2,048 | 3,500 | — |" in table  # FlashInfer's single-request call: batch 1 only
    assert "64 ×" not in table  # a skipped shape has no row
    roofline = roofline_table(m7_records, BANDWIDTH)
    assert "| 1 × 32,768 | 38.8 | 38 (15%) | 192 (77%) | 134 (54%) | 103 (41%) | 111 (44%) |" in roofline
    errors = attention_error_table(m7_records)
    assert "| 1 × 32,768 | 2.0e-05 | 2.0e-05 | 2.0e-05 | — | 1.0e-02 | 1.0e-02 |" in errors
    views = timing_views_table(m7_records)  # cold · eager (+200 µs of Python) · graph (half: warm cache)
    assert "| 1 × 32,768 | 37.0 | 3,500 · 3,700 · 1,750 | 700 · 900 · 350 | 1,000 · 1,200 · 500 |" in views
    assert "16 ×" not in views
    flush = flush_table(m7_records)  # writing the scratch buffer costs every contender the same 130 µs
    assert "| 1 × 32,768 | 3,500 · 3,630 (+130) | 700 · 830 (+130) | 1,000 · 1,130 (+130) |" in flush
    tune = tune_table(m7_records)
    assert "| 1 × 32,768 | INT4 | 512 | 4 | 512 | 350 | 10.0× | 2.0× | 11.4× |" in tune


def test_nanoserve_and_quality_tables(m7_records, m5_records):
    table = nanoserve_table(m7_records, dollars_per_hour=0.8)
    # 56 ms per token: 17.86 tokens/s → $12.44 per 1M
    assert "| 1 × 32,000 | INT4 codes + kernel 2 | 56.0 | 2.50× | 1,012 | 12.44 | 0.0100 | 100% |" in table
    assert "| 1 × 32,000 | INT4 codes, dequantize then attend | 700.0 | 0.20× |" in table
    assert "| 1 × 512 | BF16 + PyTorch attention (before M7) | 45.0 | 1.00× |" in table
    quality = quality_table(m7_records, m5_records)
    assert "| INT4 codes + kernel 2 | 0.02 | 98.0% | 20.02 | 20.00 |" in quality and "| 100% |" in quality
    assert "| Kernel 1 | 50.0 | 0.010 | 100% |" in norm_in_model_table(m7_records)
