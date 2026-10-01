"""M6 report: observables for the predictions, and the tables."""

import pytest

from fastserve.report.m6 import (
    agreement_table,
    batch_table,
    interaction_table,
    loop_table,
    lossless_table,
    m6_observables,
    one_user_table,
    parse_label,
    round_cost_table,
    step_table,
)


def test_labels_parse():
    assert parse_label("bf16-draft-k3") == ("bf16", "draft", 3)
    assert parse_label("awq-none") == ("awq", "none", None)


def test_observables(m6_records):
    obs = m6_observables(m6_records)
    assert obs["agree_all"] == pytest.approx(0.6)
    assert obs["agree_code_minus_chat"] == pytest.approx(0.2)
    assert obs["burstiness"] == pytest.approx(0.2)
    assert obs["e3_all"] == pytest.approx(1 + 0.6 + 0.36 + 0.216)
    assert obs["e3_over_iid"] == pytest.approx(1.0)  # the fixture *is* the independent formula
    assert obs["awq_drafter_agreement"] == pytest.approx(0.9)
    assert obs["fp8_target_agreement"] == pytest.approx(1.0) and obs["awq_target_agreement"] == pytest.approx(
        0.9
    )
    assert obs["verify4_cost"] == pytest.approx(1.1)
    assert obs["loop_identical"] == pytest.approx(0.75)  # the BF16 run, not the float32 control
    assert obs["lossless_worst"] == pytest.approx(50 / 84.7)
    assert obs["tv_ratio_worst"] == pytest.approx(1.0)
    assert obs["vllm_accept_vs_replay"] == pytest.approx(0.0)  # counters: 60% accepted; replay: 0.6
    assert obs["speedup_draft_k3"] == pytest.approx(1.2)
    assert obs["speedup_draft_k3_b64"] == pytest.approx(1.2 * 0.5)
    assert obs["speedup_eagle3"] == pytest.approx(1.8)
    assert obs["speedup_awq_target"] == pytest.approx(1.2)
    assert obs["vllm_identical_outputs"] == pytest.approx(0.75)  # one of four requests has another checksum
    assert obs["speedup_draftawq_over_draft"] is None  # not in the fixture: a dash, not a crash


def test_tables(m6_records):
    agreement = agreement_table(m6_records)
    assert "| Qwen3-0.6B | Code | 70.0% | 80.0% | 60.0% | 1.70 |" in agreement
    assert "| N-gram lookup | All tasks | — | — | — | 1.20 |" in agreement
    assert "| BF16 | Chat | — | — | — | 50 | 85 | 0.200 | 0.200 |" in lossless_table(m6_records)
    # every request decodes 99 tokens in 0.9 s (9.1 ms per step); 100 passes per load take 3.6 s (36 ms each)
    costs = round_cost_table(m6_records)
    assert "| BF16 | Qwen3-0.6B drafter, k = 3 | 9.1 | 36.0 | 9.0 | 0.99 | 3.96 | 1.20× |" in costs
    assert "N-gram" not in costs and "No speculation" not in costs
    steps = step_table(m6_records)
    assert "| 4 | 22.00 | 1.10× |" in steps and "| drafter, 1 token | 10.00 | 0.50× |" in steps
    one_user = one_user_table(m6_records)
    assert (
        "| Qwen3-0.6B drafter, k = 3 | 1.20× | 1.20× | 1.20× | 1.20× | 1.20× | 60% | 2.80 | 75% |" in one_user
    )
    assert "| No speculation | 1.00× |" in one_user
    batch = batch_table(m6_records)
    assert "| No speculation | — | 1.00× | 400 tok/s | 1,600 tok/s | 6,400 tok/s |" in batch
    assert "| EAGLE-3 head, k = 3 | — | 1.80× | 1.62× | 1.26× | 0.90× |" in batch
    interaction = interaction_table(m6_records)
    assert "| INT4 W4A16 (AWQ) | 54.0% |" in interaction and "| 1.20× |" in interaction
    loop = loop_table(m6_records)
    assert "| BF16 | 8 | 6 of 8 | 6 of 8 | 2.50 | 2.50 |" in loop
    assert "| float32 (control) | 8 | 8 of 8 |" in loop
