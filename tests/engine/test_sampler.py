import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.sampler import SamplingParams, sample  # noqa: E402

LOGITS = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4]]))


def test_greedy_is_argmax():
    assert sample(LOGITS, SamplingParams(temperature=0.0)).tolist() == [0, 3]


def test_sampling_is_reproducible_with_a_seeded_generator():
    params = SamplingParams(temperature=1.0)
    a = [sample(LOGITS, params, torch.Generator().manual_seed(7)).tolist() for _ in range(3)]
    assert a[0] == a[1] == a[2]


def test_top_p_keeps_only_the_nucleus():
    params = SamplingParams(temperature=1.0, top_p=0.7)  # row 0: {0.5, 0.3} reach 0.7 -> tokens 0 and 1 only
    gen = torch.Generator().manual_seed(0)
    draws = torch.stack([sample(LOGITS[:1], params, gen) for _ in range(500)]).flatten()
    assert set(draws.tolist()) == {0, 1}


def test_temperature_sampling_follows_the_distribution():
    gen = torch.Generator().manual_seed(0)
    draws = torch.stack(
        [sample(LOGITS[:1], SamplingParams(temperature=1.0), gen) for _ in range(4000)]
    ).flatten()
    freq = torch.bincount(draws, minlength=4).float() / len(draws)
    torch.testing.assert_close(freq, torch.tensor([0.5, 0.3, 0.15, 0.05]), atol=0.03, rtol=0)
