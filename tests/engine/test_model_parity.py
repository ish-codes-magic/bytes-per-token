"""nanoserve must compute exactly what Hugging Face computes (float32, tiny random models, CPU)."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("tie", [False, True])
def test_logits_match_hugging_face(make_tiny_qwen3, tie):
    hf, ours = make_tiny_qwen3(tie_word_embeddings=tie)
    ids = torch.randint(0, 256, (3, 17), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = hf(ids).logits
        got = ours(ids)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_tied_lm_head_is_the_embedding_matrix(make_tiny_qwen3):
    _, ours = make_tiny_qwen3(tie_word_embeddings=True)
    assert ours.lm_head.weight is ours.model.embed_tokens.weight


def test_select_returns_the_chosen_token_logits(tiny_model):
    ids = torch.randint(0, 256, (2, 9), generator=torch.Generator().manual_seed(2))
    select = torch.tensor([3, 8])
    with torch.no_grad():
        full = tiny_model(ids)  # [2, 9, vocab]
        picked = tiny_model(ids, select=select)  # [2, vocab]
    torch.testing.assert_close(picked, full[torch.arange(2), select])
