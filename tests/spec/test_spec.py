"""The rejection sampler, the drafters and the draft-verify loop, piece by piece."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.generate import generate  # noqa: E402
from fastserve.engine.sampler import SamplingParams  # noqa: E402
from fastserve.spec.drafters import ModelDrafter, NGramDrafter  # noqa: E402
from fastserve.spec.generate import speculative_generate  # noqa: E402
from fastserve.spec.lm import CachedLM  # noqa: E402
from fastserve.spec.rejection_sampler import verify, warp  # noqa: E402

GREEDY = SamplingParams(max_new_tokens=12)


def one_hot(tokens, vocab=4):
    return torch.nn.functional.one_hot(torch.tensor(tokens), vocab).float()


def test_warp_is_a_point_mass_when_greedy_and_a_nucleus_with_top_p():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    assert warp(logits, SamplingParams()).tolist() == [1.0, 0.0, 0.0, 0.0]
    full = warp(logits, SamplingParams(temperature=1.0))
    assert torch.allclose(full, torch.softmax(logits, dim=-1))
    nucleus = warp(logits, SamplingParams(temperature=1.0, top_p=0.7))  # top two reach 0.64 + 0.24
    assert nucleus[2:].tolist() == [0.0, 0.0] and nucleus.sum().item() == pytest.approx(1.0)


def test_greedy_verification_accepts_the_matching_prefix_and_corrects_the_first_miss():
    p = one_hot([1, 2, 3, 0])  # the target wants 1, 2, 3, then 0
    assert verify(p, one_hot([1, 2, 3]), torch.tensor([1, 2, 3])) == (3, 0)  # all right: bonus token 0
    assert verify(p, one_hot([1, 0, 3]), torch.tensor([1, 0, 3])) == (1, 2)  # wrong at index 1: target's 2
    assert verify(p, None, torch.tensor([3])) == (0, 1)  # a deterministic draft, wrong at once
    assert verify(p[:1], None, torch.tensor([], dtype=torch.long)) == (
        0,
        1,
    )  # nothing proposed: one target token


def test_identical_distributions_always_accept():
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(4, 6, generator=g), dim=-1)
    draft = torch.multinomial(p[:3], 1, generator=g).squeeze(-1)
    assert all(verify(p, p[:3], draft, g)[0] == 3 for _ in range(50))  # p/q = 1 everywhere


def test_ngram_drafter_proposes_what_followed_the_most_recent_match():
    drafter = NGramDrafter(max_n=3, min_n=2)
    #          0  1  2  3  4  5  6  7  8
    context = [7, 1, 2, 3, 9, 1, 2, 4, 1, 2]
    assert drafter.propose(context, 3) == ([4, 1, 2], None)  # "1 2" last appeared at 5–6, followed by 4 1 2
    assert drafter.propose([5, 1, 2, 3, 8, 1, 2, 3], 2) == ([8, 1], None)  # the longer match "1 2 3" wins
    assert drafter.propose([1, 2, 3, 4], 3) == ([], None)  # nothing repeats


def test_cached_lm_rolls_back_to_the_common_prefix(tiny_model):
    lm = CachedLM(tiny_model, max_len=32)
    context = list(range(10, 20))
    with torch.no_grad():
        expected = tiny_model(torch.tensor([context]))[0]
    assert torch.allclose(lm.logits_after(context, 3), expected[-3:], atol=1e-5)
    assert lm.tokens_run == 10
    lm.logits_after(context + [1, 2, 3], 1)  # three drafted tokens: only they are run
    assert lm.tokens_run == 13
    changed = context + [1, 9]  # the 2nd drafted token was rejected and replaced
    with torch.no_grad():
        expected = tiny_model(torch.tensor([changed]))[0, -1]
    assert torch.allclose(lm.logits_after(changed, 1)[0], expected, atol=1e-5)
    assert lm.tokens_run == 14  # only the replacement token was run again


@pytest.mark.parametrize("k", [1, 4])
def test_greedy_speculation_generates_exactly_the_plain_tokens(make_tiny_qwen3, k):
    target_model, draft_model = make_tiny_qwen3(seed=0)[1], make_tiny_qwen3(seed=1)[1]
    prompt = torch.randint(0, 256, (20,), generator=torch.Generator().manual_seed(2)).tolist()
    plain = generate(target_model, [prompt], GREEDY)[0]

    target = CachedLM(target_model, max_len=64)
    drafter = ModelDrafter(CachedLM(draft_model, max_len=64), GREEDY)
    result = speculative_generate(target, drafter, prompt, GREEDY, k)
    assert result.tokens == plain
    assert len(result.from_draft) == len(plain) and sum(n for _, n in result.rounds) == sum(result.from_draft)

    same = speculative_generate(  # drafting with the target itself: every token accepted, k + 1 per round
        CachedLM(target_model, max_len=64),
        ModelDrafter(CachedLM(target_model, max_len=64), GREEDY),
        prompt,
        GREEDY,
        k,
    )
    assert same.tokens == plain and all(n == proposed for proposed, n in same.rounds[:-1])
    assert len(same.rounds) == -(-len(plain) // (k + 1))  # ⌈12 / (k + 1)⌉ rounds: the last one is cut short


def test_ngram_speculation_is_also_exact_and_stops_at_a_stop_token(tiny_model):
    prompt = [5, 6, 7, 8] * 5
    params = SamplingParams(max_new_tokens=10)
    plain = generate(tiny_model, [prompt], params)[0]
    result = speculative_generate(CachedLM(tiny_model, max_len=64), NGramDrafter(), prompt, params, k=4)
    assert result.tokens == plain
    stop = SamplingParams(max_new_tokens=10, stop_token_ids=(plain[3],))
    stopped = speculative_generate(CachedLM(tiny_model, max_len=64), NGramDrafter(), prompt, stop, k=4)
    assert stopped.tokens == plain[: plain.index(plain[3]) + 1]
