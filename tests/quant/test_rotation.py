"""Rotations are orthogonal, spread outliers, and leave a folded-and-rotated model's outputs unchanged."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.quant.rotation import (  # noqa: E402
    fold_all_norms,
    hadamard,
    random_hadamard,
    rotate_model,
    rotate_residual,
)

IDS = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))


def test_hadamard_matrices_are_orthogonal():
    for n in (1, 2, 64, 1024):
        H = hadamard(n)
        assert torch.allclose(H @ H.T, torch.eye(n, dtype=H.dtype), atol=1e-12)
    R = random_hadamard(64, seed=3)
    assert torch.allclose(R @ R.T, torch.eye(64, dtype=R.dtype), atol=1e-12)
    with pytest.raises(ValueError):
        hadamard(96)


def test_a_rotation_spreads_one_huge_channel_over_all_of_them():
    x = torch.full((1, 64), 0.1, dtype=torch.float64)
    x[0, 5] = 100.0  # a "massive activation"
    peak_to_rms = lambda v: (v.abs().max() / v.pow(2).mean().sqrt()).item()  # noqa: E731
    rotated = x @ random_hadamard(64)
    assert torch.allclose(rotated.norm(), x.norm())  # same length...
    assert peak_to_rms(rotated) < peak_to_rms(x) / 4  # ...no dominant channel


@pytest.mark.parametrize("tie", [False, True])
def test_folding_norms_then_rotating_changes_nothing(make_tiny_qwen3, tie):
    _, model = make_tiny_qwen3(tie_word_embeddings=tie)
    before = model(IDS)
    fold_all_norms(model)
    assert torch.allclose(model(IDS), before, atol=1e-5)
    rotate_residual(model, random_hadamard(64, seed=1))
    assert torch.allclose(model(IDS), before, atol=1e-4)
    assert model.lm_head.weight is not model.model.embed_tokens.weight  # the head got its own copy


def test_rotating_without_folding_is_refused(make_tiny_qwen3):
    _, model = make_tiny_qwen3()
    with torch.no_grad():
        model.model.layers[0].input_layernorm.weight.mul_(1.5)
    with pytest.raises(ValueError):
        rotate_residual(model, hadamard(64))


def test_rotate_model_is_function_preserving(make_tiny_qwen3):
    _, model = make_tiny_qwen3()
    before = model(IDS)
    R = rotate_model(model, seed=2)
    assert R.shape == (64, 64)
    assert torch.allclose(model(IDS), before, atol=1e-4)
