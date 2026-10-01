"""M6's reference tasks on tiny random models (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.sampler import SamplingParams  # noqa: E402
from fastserve.experiments.m6 import agreement, agreement_bits, highlight, loop_check, lossless  # noqa: E402
from fastserve.spec.drafters import ModelDrafter, NGramDrafter  # noqa: E402
from fastserve.spec.generate import speculative_generate  # noqa: E402
from fastserve.spec.lm import CachedLM  # noqa: E402
from fastserve.spec.simulate import model_rounds  # noqa: E402

GREEDY = SamplingParams(max_new_tokens=24)


@pytest.fixture
def pair(make_tiny_qwen3):
    """A tiny target and a drafter that agrees with it some of the time: the target with noise added."""
    target = make_tiny_qwen3(seed=0)[1]
    draft = make_tiny_qwen3(seed=0)[1]
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in draft.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    return target, draft


def prompts(n, length=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (length,), generator=g).tolist() for _ in range(n)]


def test_the_replay_from_agreement_bits_equals_the_real_loop(pair):
    target, draft = pair
    for prompt in prompts(3):
        plain = speculative_generate(
            CachedLM(target, 64), ModelDrafter(CachedLM(target, 64), GREEDY), prompt, GREEDY, 0
        )
        bits = agreement_bits(draft, prompt, plain.tokens)
        assert 0 < sum(bits) < len(bits)  # the noisy drafter is right sometimes, not always
        for k in (1, 3):
            real = speculative_generate(
                CachedLM(target, 64), ModelDrafter(CachedLM(draft, 64), GREEDY), prompt, GREEDY, k
            )
            assert real.tokens == plain.tokens
            replay = model_rounds(bits, k)
            assert [a for _, a in real.rounds[:-1]] == [a for _, a in replay[: len(real.rounds) - 1]]


def test_agreement_rows_cover_every_drafter_task_and_k(pair):
    target, draft = pair
    by_task = {"chat": prompts(2, seed=1), "code": [p * 2 for p in prompts(2, length=8, seed=2)]}
    rows = agreement(target, {"small": draft, "ngram": NGramDrafter()}, by_task, GREEDY, ks=[1, 4], batch=4)
    assert {(r["drafter"], r["task"]) for r in rows} == {
        (d, t) for d in ("small", "ngram") for t in ("chat", "code", "all")
    }
    row = next(r for r in rows if r["drafter"] == "small" and r["task"] == "all")
    assert row["prompts"] == 4 and row["tokens"] == 4 * 24 and set(row["by_k"]) == {"1", "4"}
    assert 1.0 <= row["by_k"]["1"]["tokens_per_round"] <= row["by_k"]["4"]["tokens_per_round"] <= 5.0
    assert 0 < row["rates"]["agree"] < 1
    assert "rates" not in next(r for r in rows if r["drafter"] == "ngram")


def test_loop_check_and_highlight(pair):
    target, draft = pair
    check = loop_check(target, draft, prompts(2), GREEDY, k=3)
    assert (
        check["identical_outputs"] == 2
        and check["rounds_match_replay"] == 2
        and check["first_divergence"] == []
    )
    assert check["tokens_per_round_actual"] >= 1.0

    class Decoder:
        def decode(self, ids):
            return f"<{ids[0]}>"

    shown = highlight(Decoder(), draft, target, prompts(1)[0], GREEDY, k=3)
    assert len(shown["pieces"]) == len(shown["from_draft"]) == 24 and any(shown["from_draft"])


def test_lossless_measurement_passes_for_a_correct_sampler(pair):
    target, draft = pair
    out = lossless(target, draft, prompts(1)[0], samples=300, tokens=2, k=2, plain_batch=100)
    assert out["chi_square"] < out["limit"] and out["chi_square_plain"] < out["limit"]
    assert out["chi_square_own"] < out["limit"] and out["chi_square_plain_own"] < out["limit"]
    assert max(out["path_tv"].values()) < 1e-4  # float32 on CPU: the three passes agree
    assert [p["tokens"] for p in out["prefixes"]] == [1, 2]
    assert all(0 <= p["tv_spec_vs_plain"] <= 1 for p in out["prefixes"])
