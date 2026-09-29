"""Model dimensions, always read from a Hugging Face config.json and never hard-coded.

Stdlib only, so sizes (parameters, KV bytes per token) can be computed anywhere, even on the laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int  # query heads
    num_kv_heads: int  # key/value heads (fewer than query heads = grouped-query attention)
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    attention_bias: bool = False

    @classmethod
    def from_hf(cls, cfg: dict[str, Any]) -> ModelConfig:
        """Build from a config.json dict. Accepts both the transformers 4.x and 5.x layouts."""
        if cfg.get("model_type") != "qwen3":
            raise NotImplementedError(f"nanoserve implements Qwen3 only, got {cfg.get('model_type')!r}")
        # transformers 4.x writes `rope_theta` at the top level; 5.x moves it into `rope_parameters`.
        rope = cfg.get("rope_parameters") or {}
        if cfg.get("rope_scaling") or rope.get("rope_type", "default") != "default":
            raise NotImplementedError("only plain RoPE is implemented (Qwen3's default)")
        heads = cfg["num_attention_heads"]
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=heads,
            num_kv_heads=cfg.get("num_key_value_heads") or heads,
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            rope_theta=float(cfg["rope_theta"] if "rope_theta" in cfg else rope["rope_theta"]),
            rms_norm_eps=cfg["rms_norm_eps"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            attention_bias=cfg.get("attention_bias", False),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> ModelConfig:
        return cls.from_hf(json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8")))

    @property
    def gqa_group(self) -> int:
        """How many query heads share one KV head."""
        return self.num_heads // self.num_kv_heads

    def num_params(self) -> int:
        d, f, hd = self.hidden_size, self.intermediate_size, self.head_dim
        attn = d * self.num_heads * hd + 2 * d * self.num_kv_heads * hd + self.num_heads * hd * d
        attn += 2 * hd  # q_norm, k_norm
        per_layer = attn + 3 * d * f + 2 * d  # + MLP + two RMSNorms
        embed = self.vocab_size * d
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * d
        return self.num_layers * per_layer + embed + lm_head + d  # + final norm

    def kv_bytes_per_token(self, bytes_per_elem: float = 2) -> float:
        """K and V, for every layer and KV head: 2 × layers × kv_heads × head_dim × bytes."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * bytes_per_elem
