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


def test_prefill_efficiency_uses_causal_attention_flops(m2_records):
    import json
    from pathlib import Path

    from fastserve.engine.config import ModelConfig
    from fastserve.report.m2 import prefill_efficiency_table, prefill_flops

    path = Path(__file__).parents[2] / "benchmarks" / "models" / "Qwen3-0.6B.config.json"
    cfg = ModelConfig.from_hf(json.loads(path.read_text(encoding="utf-8")))
    # attention share grows with length: at 32k it's a noticeable part of the total
    assert prefill_flops(cfg, 32768) > 2 * prefill_flops(cfg, 16384)
    table = prefill_efficiency_table(m2_records, {"Qwen/Qwen3-0.6B": cfg}, peak_flops=57e12)
    assert table.count("| Qwen3-0.6B |") == 3 and "| 100 |" in table  # the fake runs use 100-token prompts


def _qwen3_small():
    import json
    from pathlib import Path

    from fastserve.engine.config import ModelConfig

    path = Path(__file__).parents[2] / "benchmarks" / "models" / "Qwen3-0.6B.config.json"
    return ModelConfig.from_hf(json.loads(path.read_text(encoding="utf-8")))


def test_plateau_uses_only_intervals_with_a_queue(m2_saturation_records):
    from fastserve.report.m2 import plateau, runs

    p = plateau(runs(m2_saturation_records, model=SMALL, workload="saturation")[0])
    assert p["seconds"] == pytest.approx(8)  # t = 1..9: both ends of each interval had requests waiting
    assert p["output_tok_s"] == pytest.approx(4000) and p["prompt_tok_s"] == pytest.approx(1000)
    assert p["step_ms"] == pytest.approx(50) and p["running"] == 200 and p["kv_usage"] == 0.5
    assert p["prompt_per_step"] == pytest.approx(50)


def test_drained_intervals_are_decode_only(m2_saturation_records):
    from fastserve.report.m2 import drained, plateau, runs

    p = plateau(runs(m2_saturation_records, model=SMALL, workload="saturation")[0], drained)
    assert p["seconds"] == pytest.approx(4) and p["prompt_per_step"] == 0
    assert p["running"] == 100 and p["step_ms"] == pytest.approx(25)


def test_saturation_table_compares_throughput_with_streaming_the_bytes(m2_saturation_records):
    from fastserve.report.m2 import saturation_table

    cfg = _qwen3_small()
    table = saturation_table(m2_saturation_records, {SMALL: cfg}, bandwidth=262e9)
    # half of 100,000 cached tokens across 200 sequences; each step streams the weights plus that KV
    step_s = (2 * cfg.num_params() + 50_000 * cfg.kv_bytes_per_token()) / 262e9
    ratio = 4000 / (200 / step_s)
    # token-weighted context of 256-token prompts with 192-token answers: 256 + 191/2
    assert "| Qwen3-0.6B | 8 | 200 | 352 / 250 | 50% | 4,000 |" in table
    assert table.rstrip().endswith(f"| {ratio:.0%} |")


def test_decode_efficiency_puts_one_sequence_next_to_the_saturated_engine(m2_records, m2_saturation_records):
    from fastserve.report.m2 import decode_efficiency_table

    table = decode_efficiency_table(m2_records, m2_saturation_records, {SMALL: _qwen3_small()}, 262e9)
    assert "| one user, chat | 1 | 124 | 0 |" in table  # 100-token prompts + half of 50 output tokens
    assert "| saturated, queue waiting | 200 | 250 | 50 |" in table
    assert "| saturated, queue drained | 100 | 250 | 0 |" in table


def test_peak_definition_contrasts_the_sweep_average_with_the_plateau(m2_records, m2_saturation_records):
    from fastserve.report.m2 import mean_decoding, peak_definition_table, runs

    sweep = runs(m2_records, model=SMALL, workload="throughput", mode="open")[0]
    # 10 requests, 0.1 s apart, each decoding for 1 s: at most 10 overlap, fewer during ramp-up and drain
    assert 0 < mean_decoding(sweep["requests"]) < 10
    table = peak_definition_table(m2_records, m2_saturation_records)
    assert "| Qwen3-0.6B | 4,000 (64 users) |" in table and table.splitlines()[2].endswith("| 4,000 | 200 |")


def test_prefill_budget_predicts_the_batch_with_littles_law(m2_saturation_records):
    from fastserve.report.m2 import prefill_budget_table

    table = prefill_budget_table(m2_saturation_records)
    # a 100 ms step carries 1,988 prompt + 60 output tokens: R = B·O / (P + O) = 2048 × 64 / 2176 ≈ 60
    assert "| Qwen3-0.6B | 2,048 | 2,112 + 64 | 60 | 60 | 100 | 9.4 |" in table


def test_repeat_table_pairs_identical_load_points(m2_records):
    import copy

    from fastserve.report.m2 import repeat_table

    again = copy.deepcopy(m2_records)
    for r in again:
        if r["experiment"] == "serving" and r["metrics"]["workload"] == "shared_prefix":
            r["metrics"]["summary"]["output_throughput"] *= 1.1
    table = repeat_table(m2_records, [r for r in again if r["metrics"].get("workload") == "shared_prefix"])
    assert table.count("| shared_prefix, 16 req/s |") == 2  # one row per model, nothing else matched
    assert "| 900 / 990 |" in table and table.rstrip().endswith("| 10% |")
