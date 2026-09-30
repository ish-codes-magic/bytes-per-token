"""M3 report: observables for the predictions, and readable tables."""

import pytest

from fastserve.report.m3 import (
    config_table,
    configs,
    label,
    m3_observables,
    sensitivity_shares,
    sensitivity_table,
    single,
)


def test_the_newest_record_of_each_configuration_wins(m3_records):
    assert configs(m3_records)["rtn-int4-g128"]["mean_kl"] == 0.1  # not the stale 9.9


def test_observables_are_the_predicted_ratios(m3_records):
    obs = m3_observables(m3_records)
    assert obs["bf16_ppl"] == 20.0 and obs["kl_int4_g128"] == 0.1
    assert obs["channel_over_g128"] == pytest.approx(5)
    assert obs["gptq_over_rtn"] == pytest.approx(0.5) and obs["awq_over_rtn"] == pytest.approx(0.6)
    assert obs["gptq_over_library"] == pytest.approx(0.9) and obs["awq_over_library"] == pytest.approx(0.9)
    assert obs["smoothquant_gain"] == pytest.approx(0.1) and obs["outlier_ratio"] == 1500
    assert obs["calib_8_over_128"] == pytest.approx(1.2) and obs["calib_wiki_over_c4"] == pytest.approx(0.9)
    assert obs["large_over_small_int4"] == pytest.approx(0.5)
    assert obs["down_proj_share"] == pytest.approx(100 * 4 / 10)  # 4 of every 10 units of KL
    assert obs["edge_layer_share"] == pytest.approx(100 * 3 / 28)


def test_labels_read_like_the_method():
    assert label({"method": "rtn", "bits": 4}) == "RTN INT4 g128"
    assert label({"method": "rtn", "bits": 4, "granularity": "channel", "rotate": True}) == (
        "rotated, RTN INT4 per-channel"
    )
    assert label({"method": "gptq", "bits": 4, "full_range": True, "calibration": "code"}) == (
        "GPTQ INT4 g128, full range, calibrated on code"
    )
    assert label({"method": "awq", "bits": 3, "clip": True}) == "AWQ INT3 g128 (duo scaling, clip search)"
    assert label({"method": "w8a8", "format": "int8", "act": "tensor", "static": True, "smooth": 0.5}) == (
        "SmoothQuant + W8A8 INT8, static per-tensor activations"
    )
    assert label({"method": "library", "checkpoint": "Qwen3-0.6B-gptq", "bits": 4, "full_range": True}) == (
        "llm-compressor GPTQ INT4 g128, full range"
    )


def test_config_table_has_one_row_per_known_configuration(m3_records):
    table = config_table(m3_records, ["bf16", "rtn-int4-g128", "missing"], reference="rtn-int4-g128")
    assert table.count("\n") == 3  # header, separator, two rows
    assert "| BF16 (reference) |" in table and "| 1.00× |" in table and "+10.0%" in table


def test_sensitivity_summaries(m3_records):
    scan = single(m3_records, "m3_sensitivity")
    assert sensitivity_shares(scan, 28)["down_proj"] == pytest.approx(40)
    table = sensitivity_table(m3_records)
    assert table.startswith("| Module type |") and "| down_proj |" in table and "40%" in table
