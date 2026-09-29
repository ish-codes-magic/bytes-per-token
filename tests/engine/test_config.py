import pytest

from fastserve.engine.config import ModelConfig

# The real Qwen/Qwen3-0.6B config.json (transformers 4.x layout: rope_theta at the top level).
QWEN3_0_6B = {
    "model_type": "qwen3",
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rope_theta": 1000000,
    "rope_scaling": None,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": True,
    "attention_bias": False,
}


def test_reads_the_real_qwen3_config():
    cfg = ModelConfig.from_hf(QWEN3_0_6B)
    assert (cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (28, 16, 8, 128)
    assert cfg.gqa_group == 2
    assert cfg.rope_theta == 1e6
    assert cfg.kv_bytes_per_token() == 114_688  # 2 × 28 × 8 × 128 × 2 bytes = 112 KiB
    assert cfg.num_params() == pytest.approx(0.596e9, rel=1e-3)


def test_reads_the_transformers_5_rope_layout():
    v5 = {k: v for k, v in QWEN3_0_6B.items() if k not in ("rope_theta", "rope_scaling")}
    v5["rope_parameters"] = {"rope_type": "default", "rope_theta": 1000000.0}
    assert ModelConfig.from_hf(v5).rope_theta == 1e6


@pytest.mark.parametrize(
    "change",
    [{"model_type": "llama"}, {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}],
)
def test_refuses_what_it_does_not_implement(change):
    with pytest.raises(NotImplementedError):
        ModelConfig.from_hf({**QWEN3_0_6B, **change})


@pytest.mark.parametrize("tie", [False, True])
def test_param_count_matches_hugging_face(make_tiny_qwen3, tie):
    hf, ours = make_tiny_qwen3(tie_word_embeddings=tie)
    assert ours.config.num_params() == sum(p.numel() for p in hf.parameters())  # tied weights count once
