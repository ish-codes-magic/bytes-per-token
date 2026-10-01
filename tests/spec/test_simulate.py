import pytest

from fastserve.spec.simulate import (
    expected_tokens,
    from_draft,
    model_rounds,
    ngram_rounds,
    run_rates,
    summarize,
)

T, F = True, False


def test_model_rounds_accept_runs_of_agreement_up_to_k():
    agree = [T, T, F, T, T, T, T, F]
    #  k=2: [T T] + F(target) | [T T] + T(target) | [T] then F is the target's
    assert model_rounds(agree, 2) == [(2, 2), (2, 2), (2, 1)]
    assert model_rounds(agree, 1) == [(1, 1), (1, 0), (1, 1), (1, 1), (1, 0)]
    assert (
        model_rounds([F, F, F], 4) == [(4, 0)] * 3
    )  # nothing accepted: one token per round, as plain decoding
    assert model_rounds([T] * 10, 4) == [(4, 4), (4, 4)]  # everything accepted: k + 1 tokens per round


def test_from_draft_marks_accepted_tokens():
    rounds = model_rounds([T, T, F, T, F], 2)
    assert rounds == [(2, 2), (2, 1)]
    assert from_draft(rounds, 5) == [T, T, F, T, F]


def test_summary_and_theory():
    stats = summarize([[(2, 2), (2, 2), (2, 1)]], tokens=8)
    assert stats["tokens_per_round"] == pytest.approx(8 / 3)
    assert stats["acceptance_rate"] == pytest.approx(5 / 6)
    assert stats["accepted_histogram"] == [0, 1, 2]
    assert expected_tokens(0.5, 2) == pytest.approx(1 + 0.5 + 0.25)
    assert expected_tokens(1.0, 3) == 4.0 and expected_tokens(0.0, 3) == 1.0


def test_run_rates_show_burstiness():
    bursty = [T, T, T, T, F, F, F, F] * 4
    rates = run_rates(bursty)
    assert rates["agree"] == 0.5 and rates["after_agree"] > 0.7 and rates["after_miss"] < 0.3


def test_ngram_rounds_copy_from_the_prompt():
    class Lookup:  # proposes the tokens after the last earlier occurrence of the final token
        def propose(self, context, k):
            last = context[-1]
            for start in range(len(context) - 2, -1, -1):
                if context[start] == last:
                    return context[start + 1 : start + 1 + k], None
            return [], None

    prompt, output = [1, 2, 3, 4, 9], [1, 2, 3, 7]
    # round 1: nothing follows 9 earlier → no draft, the target writes 1. round 2: after "1" came 2 3 4:
    # 2 and 3 are right, 4 is wrong (the target writes 7).
    assert ngram_rounds(prompt, output, 3, Lookup()) == [(0, 0), (3, 2)]
