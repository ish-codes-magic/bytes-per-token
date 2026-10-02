"""M8 report: the ablation tables, the frozen predictions against measurements, and the observables."""

import pytest

from fastserve.report import m8

LARGE = "Qwen/Qwen3-1.7B"
FROZEN = {
    "predictions": [
        {"model": LARGE, "label": "wkps", "workload": "m8_latency", "tok_s": 237.6},  # measured 216: +10%
        {"model": LARGE, "label": "base", "workload": "m8_latency", "tok_s": 80.0},  # measured 100: −20%
        {"model": LARGE, "label": "w", "workload": "capacity", "tok_s": 312.5},  # exact
        {"model": LARGE, "label": "zz", "workload": "capacity", "tok_s": 1.0},  # never ran: skipped
    ]
}


def test_labels_and_basic_accessors(m8_records):
    assert m8.without("wkps", "k") == "wps" and m8.without("w", "w") == "base"
    assert m8.label_pair("s", "w") == "ws" and len(m8.pairs()) == 6
    assert m8.tok_s(m8_records, LARGE, "base", "m8_latency") == 100.0
    assert m8.speedup(m8_records, LARGE, "wkps", "m8_latency") == pytest.approx(1.5 * 1.6 * 0.9)
    assert m8.speedup(m8_records, LARGE, "nope", "m8_latency") is None
    assert m8.tokens_per_joule(m8_records, LARGE, "base", "m8_latency") == pytest.approx(100 / 70)
    assert m8.dollars(100.0, 0.8) == pytest.approx(0.8 / 0.36) and m8.dollars(None, 0.8) is None


def test_ladder_and_steps(m8_records):
    ladder = m8.ladder_table(m8_records, LARGE, dollars_per_hour=0.8)
    assert "| Latency (1 user, real prompts) | 100 | 2.22 | 1.50× | 1.50× | 1.50× | 2.16× | 1.03 |" in ladder
    assert "| Multi-turn (8 users, shared prefixes) | 200 | 1.11 | 1.25× | 1.38× | 2.75× | 2.97× |" in ladder
    steps = m8.step_table(m8_records, LARGE, "capacity", dollars_per_hour=0.8)
    assert "| stock BF16 | `base` | 250 | 1.00× | — | 384.0 | 50 | 0.89 | 3.6 | 150,000 |" in steps
    assert "| + FP8 weights | `w` | 312 | 1.25× | 1.25× |" in steps
    assert "| (control: + FlashInfer, BF16 cache) | `wf` | 344 | 1.38× | — |" in steps  # not a ladder step
    assert "| + FP8 KV cache | `wk` | 500 | 2.00× | 1.60× |" in steps  # against `w`, not the control
    assert "| + speculative decoding | `wkps` | 540 | 2.16× | 1.08× |" in steps and "300,000 |" in steps
    assert "| (branch: INT4 weights instead of FP8) | `akps` | 576 | 2.30× | — |" in steps


def test_alone_is_not_in_the_stack_when_techniques_compete(m8_records):
    alone, in_stack = m8.alone_and_in_stack(m8_records, LARGE, "w", "m8_latency")
    assert alone == pytest.approx(1.5) and in_stack == pytest.approx(1.35)  # wkps ÷ kps = 1.5 × 0.9
    table = m8.leave_one_out_table(m8_records, LARGE)
    assert (
        "| Latency (1 user, real prompts) | 1.50× · 1.35× | 1.00× · 1.00× | 1.00× · 1.00× | 1.60× · 1.44× |"
        in table
    )
    assert (
        "| Multi-turn (8 users, shared prefixes) | 1.25× · 1.12× | 1.10× · 1.10× | 2.00× · 2.00× |" in table
    )


def test_interactions_find_the_one_competing_pair(m8_records):
    assert m8.interaction(m8_records, LARGE, "w", "s", "m8_latency") == pytest.approx(0.9)
    assert m8.interaction(m8_records, LARGE, "k", "p", "multi_turn") == pytest.approx(1.0)
    table = m8.interaction_table(m8_records, LARGE)
    assert "| FP8 weights + speculative decoding | `ws` | 0.90 | 0.90 | 0.90 | 0.90 |" in table
    assert "| FP8 KV cache + prefix caching | `kp` | 1.00 | 1.00 | 1.00 | 1.00 |" in table


def test_repeats_energy_and_failures(m8_records):
    diffs = {(label, w): d for label, w, d in m8.repeat_differences(m8_records, LARGE)}
    assert diffs[("base", "m8_latency")] == pytest.approx(0.02) and diffs[
        ("wkps", "capacity")
    ] == pytest.approx(-0.03)
    assert ("base", "long_32k") not in diffs  # the repeats did not run the ladder-only workload
    assert "| `wkps` | Capacity (96 users, 4k-token prompts) | -3.0% |" in m8.repeat_table(m8_records, LARGE)
    assert "| Latency (1 user, real prompts) | 70 | 70 | 1.43 | 3.09 | 2.16× |" in m8.energy_table(
        m8_records, LARGE
    )
    assert "| `0.6b-wks` | RuntimeError: vLLM exited during startup |" in m8.failures_table(m8_records)
    assert m8.failures_table([]) == "*Every server of the plan ran.*"
    kept = m8.kept_table(m8_records, LARGE)
    assert "| `s` | `ws` | `ks` | `ps` | `wkps` | `akps` |" in kept and "| 2.00 | 2.00 |" in kept


def test_frozen_predictions_are_scored_against_what_ran(m8_records):
    errors = {(r["label"], r["workload"]): r["error"] for r in m8.prediction_errors(m8_records, FROZEN)}
    assert errors == {
        ("wkps", "m8_latency"): pytest.approx(0.10),
        ("base", "m8_latency"): pytest.approx(-0.20),
        ("w", "capacity"): pytest.approx(0.0),
    }
    table = m8.model_error_table(m8_records, FROZEN)
    assert "| Latency (1 user, real prompts) | 2 | 15.0% |" in table
    assert "| Servers with speculation | 1 | 10.0% | 10.0% | 10% | 100% |" in table
    assert "| **All** | 3 | 10.0% | 0.0% | 20% | 67% |" in table
    worst = m8.worst_predictions(m8_records, FROZEN, n=1)
    assert "| Qwen3-1.7B | `base` | Latency (1 user, real prompts) | 100 | 80 | -20% |" in worst


def test_cost_and_quality(m8_records, m5_records, m4_records):
    m2_quality = []  # no BF16 baseline scores in this test: those cells are dashes
    cost = m8.cost_table(m8_records, dollars_per_hour=0.8)
    row = (
        "| Qwen3-1.7B | Multi-turn (8 users, shared prefixes) | 1.11 | 0.37 | 2.97× | `wkps` | 0.37 | 2.97× |"
    )
    assert row in cost
    quality = m8.quality_table(m8_records, m5_records, m4_records, m2_quality)
    assert "| Qwen3-1.7B | FP8 weights + FP8 KV (`wk`) | 20.40 | 100% | 50.0 | 60.0 | 30.0 |" in quality
    assert (
        "| Qwen3-0.6B | FP8 weights + FP8 KV (`wk`) | — | — | — | — | — |" in quality
    )  # not measured: dashes


def test_the_best_stack_is_chosen_among_deployable_servers(m8_records):
    assert "fs" not in m8.candidates(m8_records, LARGE) and "g" not in m8.candidates(m8_records, LARGE)
    assert "aps" in m8.candidates(m8_records, LARGE)
    assert "aps" not in m8.candidates(m8_records, LARGE, allow_int4=False)
    assert all(not label.endswith("-r2") for label in m8.candidates(m8_records, LARGE))
    label, rate = m8.best(m8_records, LARGE, "multi_turn", allow_int4=False)
    assert label == "wkps" and rate == pytest.approx(200 * 1.25 * 1.1 * 2.0 * 1.2 * 0.9)
    assert m8.best(m8_records, LARGE, "multi_turn")[0] == "akps"  # INT4 does not compete with speculation
    assert m8.best(m8_records, "no/model", "multi_turn") is None
    table = m8.best_table(m8_records, LARGE, {"": 20.0, "w": 20.2, "wk": 20.4})
    assert (
        "| Multi-turn (8 users, shared prefixes) | 200 | 2.97× | `wkps` | 2.97× | +2.0% | `akps` | 3.17× |"
        in table
    )
    assert "| Long (1 user, 32k tokens) | 10 | 1.62× | `wkps` | 1.62× | +2.0% | `akps` | 1.73× |" in table
    assert "| `wkps` | 1.62× | — | `akps` |" in m8.best_table(m8_records, LARGE)  # no perplexity given


def test_the_collision_table_puts_controls_next_to_what_they_explain(m8_records):
    assert m8.graph_mode(m8.graphs_of(m8_records, LARGE, "base")) == "full"
    assert m8.graph_mode(m8.graphs_of(m8_records, LARGE, "ks")) == "piecewise"
    assert m8.graph_mode(None) == "—"
    assert m8.fallback_message(m8_records, LARGE, "fs").startswith("CUDAGraphMode.FULL_AND_PIECEWISE")
    assert m8.fallback_message(m8_records, LARGE, "s") is None
    assert m8.pass_ms(m8_records, LARGE, "base") == pytest.approx(10.0)
    assert m8.pass_ms(m8_records, LARGE, "fs") == pytest.approx(2 * 1e3 / 120)  # two tokens per pass
    table = m8.collision_table(m8_records, LARGE)
    assert "| `base` | stock | FLASH_ATTN | full | 100 | 1.00 | 10.0 | 70 | 2,000 |" in table
    name = "speculation on FlashInfer, BF16 cache (control)"
    assert f"| `fs` | {name} | FLASHINFER | piecewise | 120 | 2.00 | 16.7 | 70 | 1,800 |" in table
    assert (
        "| `sg` | speculation, piecewise graphs only (control) | FLASH_ATTN | piecewise | 144 | 2.00 | 13.9 |"
        in table
    )
    observed = m8.control_observables(m8_records)
    assert observed["control_fs_vs_ks_latency"] == pytest.approx(0.75)
    assert observed["control_sg_latency"] == pytest.approx(0.9)
    assert observed["control_g_busy"] == pytest.approx(0.9)
    assert observed["control_aps_latency"] == pytest.approx(3.2)
    assert observed["control_aps_multi_turn"] == pytest.approx(1.2 * 2.0 * 1.2)


def test_observables(m8_records, m4_records):
    obs = m8.m8_observables(m8_records, FROZEN, m4_records, [])
    assert obs["full_latency"] == pytest.approx(2.16) and obs["full_capacity"] == pytest.approx(2.16)
    assert obs["full_multi_turn"] == pytest.approx(1.25 * 1.1 * 2.0 * 1.2 * 0.9)
    assert obs["int4_latency"] == pytest.approx(320 / 216)  # INT4 does not compete with speculation here
    assert obs["interaction_ws_latency"] == pytest.approx(0.9)
    assert obs["interaction_ks_capacity"] == pytest.approx(1.0)
    assert obs["loo_s_latency"] == pytest.approx(1.44) and obs["loo_w_latency"] == pytest.approx(1.35)
    assert obs["loo_k_capacity"] == pytest.approx(1.6) and obs["loo_p_multi_turn"] == pytest.approx(2.0)
    assert obs["prefix_without_prefixes"] == pytest.approx(1.0)
    assert obs["flashinfer_share_capacity"] == pytest.approx(0.125 / 0.75)  # (1.375 − 1.25) ÷ (2.0 − 1.25)
    assert obs["eagle_kept_capacity"] == pytest.approx(2.0)
    assert obs["model_median_error"] == pytest.approx(0.10) and obs["model_within_15"] == pytest.approx(2 / 3)
    assert obs["model_median_error_plain"] == pytest.approx(0.10)  # base −20%, w exact: the median of two
    assert obs["repeat_difference"] == pytest.approx(0.03)
    assert obs["energy_full_latency"] == pytest.approx(2.16) and obs["power_busy_base"] == 70.0
    assert obs["small_full_latency"] is None  # the small model is not in this fixture
    assert obs["quality_needle"] == 1.0
