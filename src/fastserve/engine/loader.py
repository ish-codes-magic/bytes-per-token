"""Load weights into nanoserve: from a Hugging Face checkpoint directory, or from an in-memory state dict."""

from __future__ import annotations

from pathlib import Path

import torch

from fastserve.engine.config import ModelConfig
from fastserve.engine.model import CausalLM


def from_state_dict(
    cfg: ModelConfig, state: dict[str, torch.Tensor], *, device: torch.device | str, dtype: torch.dtype
) -> CausalLM:
    """Build the model directly on `device` from a state dict with Hugging Face parameter names.

    The model is created on the "meta" device (shapes only, no memory), then `assign=True` makes each
    parameter *be* the loaded tensor, so no second copy of the weights is ever allocated.
    """
    with torch.device("meta"):
        model = CausalLM(cfg)
    state = {name: tensor.to(device=device, dtype=dtype) for name, tensor in state.items()}
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)  # tied: it *is* the embedding matrix
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [name for name in missing if not (cfg.tie_word_embeddings and name == "lm_head.weight")]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_weights()  # assign=True replaced the embedding Parameter, so re-point the LM head at it
    return model.eval()


def load_pretrained(
    model_dir: str | Path, *, device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> CausalLM:
    """Load config.json + *.safetensors from a downloaded Hugging Face model directory."""
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        state.update(load_file(file, device=str(device)))
    return from_state_dict(ModelConfig.from_pretrained(model_dir), state, device=device, dtype=dtype)
