import pytest

torch = pytest.importorskip("torch")


def test_fused_attention_matches_the_reference(tiny_model):
    from fastserve.engine.model import use_fused_attention

    ids = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = tiny_model(ids)
        fused = use_fused_attention(tiny_model)(ids)
    use_fused_attention(tiny_model, False)
    assert torch.allclose(fused, ref, atol=1e-5)
