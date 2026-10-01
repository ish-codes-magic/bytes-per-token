"""Speculative decoding must not change the output distribution. Two checks, both against an EXACT target
distribution (no second sample to compare with, so the only noise is the speculative sample's own):

1. A toy bigram "model" with a 5-token vocabulary: the probability of every 3-token output is a product of
   table entries. Thousands of speculative samples must match it (chi-square).
2. Tiny random Qwen3 models (a target and a different drafter): the first token's distribution is the
   target's softmax, and the second token's marginal is computed by running the target on every possible
   first token.

Each check also runs a *negative control*: a drafter that lies about the distribution it drew from, so that
every draft token gets accepted. The same test must reject it, which shows the test can tell a correct
sampler from a broken one.
"""

from collections import Counter
from itertools import product

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.sampler import SamplingParams  # noqa: E402
from fastserve.spec.drafters import ModelDrafter  # noqa: E402
from fastserve.spec.generate import speculative_generate  # noqa: E402
from fastserve.spec.lm import CachedLM, TableLM  # noqa: E402
from fastserve.spec.stats import chi_square, chi_square_limit  # noqa: E402

SAMPLE = SamplingParams(max_new_tokens=3, temperature=1.0)


def random_table(vocab: int, seed: int) -> torch.Tensor:
    """A [vocab, vocab] bigram table: row t is the distribution of the token after t."""
    return torch.softmax(
        2.0 * torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(seed)), dim=-1
    )


class FixedDrafter:
    """A deterministic drafter (like n-gram lookup): always proposes the same tokens, with no distribution."""

    def __init__(self, tokens):
        self.tokens = tokens

    def propose(self, context, k, generator=None):
        return self.tokens[:k], None


class LyingDrafter(ModelDrafter):
    """Draws from its model but reports q ≈ 0, so every draft token is accepted: the output then follows the
    DRAFTER's distribution, which is what a sampler without the rejection step would produce."""

    def propose(self, context, k, generator=None):
        tokens, q = super().propose(context, k, generator)
        return tokens, torch.full_like(q, 1e-9)


def toy_statistic(drafter, target_table, samples: int, k: int) -> tuple[float, int]:
    """Chi-square of `samples` speculative 3-token outputs against the bigram target's exact distribution."""
    vocab, start = target_table.shape[0], 0
    target, generator = TableLM(target_table), torch.Generator().manual_seed(0)
    counts = Counter(
        tuple(speculative_generate(target, drafter, [start], SAMPLE, k, generator).tokens)
        for _ in range(samples)
    )
    exact = {
        (a, b, c): float(target_table[start, a] * target_table[a, b] * target_table[b, c])
        for a, b, c in product(range(vocab), repeat=3)
    }
    return chi_square(counts, exact)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_toy_model_drafter_is_lossless(k):
    target, draft = random_table(5, seed=1), random_table(5, seed=2)
    statistic, dof = toy_statistic(ModelDrafter(TableLM(draft), SAMPLE), target, samples=6000, k=k)
    assert statistic < chi_square_limit(dof), (statistic, dof)


def test_toy_deterministic_drafter_is_lossless():
    statistic, dof = toy_statistic(FixedDrafter([3, 1, 4]), random_table(5, seed=1), samples=6000, k=3)
    assert statistic < chi_square_limit(dof), (statistic, dof)


def test_toy_negative_control_a_lying_drafter_is_caught():
    target, draft = random_table(5, seed=1), random_table(5, seed=2)
    statistic, dof = toy_statistic(LyingDrafter(TableLM(draft), SAMPLE), target, samples=6000, k=2)
    assert statistic > 3 * chi_square_limit(dof), (statistic, dof)


# ---- real (tiny) models -----------------------------------------------------------------------------------


def exact_marginals(model, prompt: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's exact distribution of the 1st generated token, and the 2nd's marginal: [vocab] each."""
    with torch.no_grad():
        first = torch.softmax(model(torch.tensor([prompt]))[0, -1].float(), dim=-1)  # [vocab]
        vocab = first.shape[0]
        extended = torch.tensor([[*prompt, t] for t in range(vocab)])  # every possible first token
        second_given_first = torch.softmax(model(extended)[:, -1].float(), dim=-1)  # [vocab, vocab]
    return first, first @ second_given_first


def model_statistics(target_model, drafter, prompt, samples: int, k: int):
    target = CachedLM(target_model, max_len=len(prompt) + 8)
    generator, params = torch.Generator().manual_seed(0), SamplingParams(max_new_tokens=2, temperature=1.0)
    outputs = [
        speculative_generate(target, drafter, prompt, params, k, generator).tokens for _ in range(samples)
    ]
    first, second = exact_marginals(target_model, prompt)
    return [
        chi_square(Counter(o[i] for o in outputs), dict(enumerate(dist.tolist())))
        for i, dist in enumerate((first, second))
    ]


def test_tiny_models_are_lossless_and_a_lying_drafter_is_caught(make_tiny_qwen3):
    target_model, draft_model = make_tiny_qwen3(seed=0)[1], make_tiny_qwen3(seed=1)[1]
    prompt = torch.randint(0, 256, (12,), generator=torch.Generator().manual_seed(5)).tolist()
    params = SamplingParams(temperature=1.0)

    honest = ModelDrafter(CachedLM(draft_model, max_len=len(prompt) + 8), params)
    for statistic, dof in model_statistics(target_model, honest, prompt, samples=1000, k=2):
        assert statistic < chi_square_limit(dof), (statistic, dof)

    lying = LyingDrafter(CachedLM(draft_model, max_len=len(prompt) + 8), params)
    statistic, dof = model_statistics(target_model, lying, prompt, samples=1000, k=2)[0]
    assert statistic > chi_square_limit(dof), (statistic, dof)
