import math

import pytest

torch = pytest.importorskip("torch")

from fastserve.quality.perplexity import evaluate, token_windows  # noqa: E402


def test_token_windows_are_non_overlapping():
    windows = token_windows(list(range(25)), window=8, max_windows=10)
    assert windows.shape == (3, 8) and windows[1, 0].item() == 8


def test_a_model_against_itself_has_zero_kl(make_tiny_qwen3):
    hf, ours = make_tiny_qwen3()
    windows = torch.randint(0, 256, (3, 16), generator=torch.Generator().manual_seed(0))
    hf_logits = lambda ids: hf(ids).logits  # noqa: E731
    same = evaluate(windows, hf_logits, hf_logits, batch=2)
    assert same["mean_kl"] == pytest.approx(0, abs=1e-6) and same["top1_agreement"] == 1.0
    # nanoserve computes the same function, so it too is ~zero KL from Hugging Face.
    assert evaluate(windows, hf_logits, ours)["mean_kl"] < 1e-8


def test_perplexity_is_exp_of_mean_negative_log_likelihood(make_tiny_qwen3):
    _, ours = make_tiny_qwen3()
    windows = torch.randint(0, 256, (2, 10), generator=torch.Generator().manual_seed(1))
    logp = torch.log_softmax(ours(windows)[:, :-1].float(), -1)
    nll = -logp.gather(-1, windows[:, 1:, None]).mean().item()
    assert evaluate(windows, ours)["perplexity_ref"] == pytest.approx(math.exp(nll), rel=1e-5)
