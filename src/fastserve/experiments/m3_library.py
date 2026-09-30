"""llm-compressor's GPTQ and AWQ on our model and calibration tokens: the reference to check ours against.

The library runs in its own image (infra/quant.lock). Its result is saved as a dense BF16 state dict with
every weight already rounded, under Hugging Face's parameter names, so nanoserve loads and scores it exactly
like our own quantized models.

Settings (llm-compressor 0.14.0 defaults, read from the installed source):
- GPTQ: the W4A16 scheme (INT4, symmetric, group 128, scale = max|w| / 7.5), act-order "static", block 128,
  dampening 0.01
- AWQ: the AWQ transform (duo scaling, 20-point grid, loss on each parent module's output), then
  round-to-nearest with the same W4A16 scheme
"""

from __future__ import annotations

import time
from typing import Any

LIBRARY_DIR = "/cache/m3"  # where the checkpoints go: on the Modal Volume, shared with the research image
QUANT_PARAMS = ("weight_scale", "weight_zero_point", "weight_g_idx", "weight_global_scale")


def library_checkpoint(method: str, model_name: str, calib: dict[str, Any], out_path: str) -> dict[str, Any]:
    """Quantize with llm-compressor, save the dense rounded weights to out_path, and describe the result."""
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform import AWQModifier
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fastserve.engine.loader import model_dir
    from fastserve.quality.text import calibration_ids

    path = model_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = calibration_ids(tokenizer, calib["source"], calib["samples"], calib["seq_len"])
    dataset = Dataset.from_dict({"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()})
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)

    scheme = {"targets": ["Linear"], "scheme": "W4A16", "ignore": ["lm_head"]}
    if method == "gptq":
        recipe = [GPTQModifier(**scheme)]
    elif method == "awq":
        recipe = [
            AWQModifier(),
            QuantizationModifier(**scheme),
        ]  # what the deprecated AWQModifier(...) builds
    else:
        raise ValueError(f"unknown library method {method!r}")

    start = time.perf_counter()
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=calib["seq_len"],
        num_calibration_samples=calib["samples"],
    )
    seconds = time.perf_counter() - start
    state, levels = dense_rounded_state(model)
    save_file(state, out_path)
    return {
        "method": method,
        "model": model_name,
        "checkpoint": out_path,
        "quantize_s": seconds,
        "tensors": len(state),
        "max_levels_per_group": max(levels.values()),  # ≤ 16 for INT4: the weights really are on the grid
    }


def dense_rounded_state(model: Any) -> tuple[dict[str, Any], dict[str, int]]:
    """Every parameter nanoserve needs, with quantized weights replaced by their rounded (dequantized) values.

    Also returns, per quantized linear, the largest number of distinct values in any group of 128 weights.
    """
    import torch
    from compressed_tensors.quantization.lifecycle.forward import forward_quantize

    rounded: dict[str, torch.Tensor] = {}
    levels: dict[str, int] = {}
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None or not hasattr(module, "weight_scale"):
            continue
        with torch.no_grad():
            w = forward_quantize(module, module.weight, "weight", scheme.weights).detach()
        rounded[f"{name}.weight"] = w
        levels[name] = max(len(torch.unique(g)) for g in w[:8].float().reshape(-1, 128))
    state = {}
    for key, tensor in model.state_dict().items():
        if key.rsplit(".", 1)[-1] in QUANT_PARAMS:
            continue
        if key == "lm_head.weight" and model.config.tie_word_embeddings:
            continue  # tied to the embedding; nanoserve re-ties it on load
        state[key] = rounded.get(key, tensor).to(torch.bfloat16).contiguous().cpu()
    return state, levels
