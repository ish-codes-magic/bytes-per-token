"""M4 checkpoints: llm-compressor turns each model into the formats vLLM can serve with real low-bit kernels.

Four formats, all with embeddings and the LM head left in BF16 (as in M3):

    fp8     W8A8 FP8: weights FP8 per output channel, activations FP8 per token (dynamic)   no calibration
    int8    W8A8 INT8: SmoothQuant (strength 0.8), then GPTQ, INT8 per channel / per token   calibration
    gptq    W4A16: GPTQ, INT4 groups of 128, activations stay 16-bit                         calibration
    awq     W4A16: AWQ scaling, then round-to-nearest to the same grid                       calibration

Calibration uses the M3 set (C4, 128 × 2,048 tokens). Each checkpoint is written twice to the Modal Volume:
- compressed, with its tokenizer: what vLLM loads
- as a dense BF16 state dict of the rounded weights: what nanoserve scores for KL, exactly like M3
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

CHECKPOINT_DIR = "/cache/m4"
FORMATS = ("fp8", "int8", "gptq", "awq")


def recipe(fmt: str) -> list:
    """The llm-compressor recipe for one format (the installed 0.14.0's presets, read before use)."""
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform import AWQModifier, SmoothQuantModifier

    ignore = ["lm_head"]
    if fmt == "fp8":
        return [QuantizationModifier(targets=["Linear"], scheme="FP8_DYNAMIC", ignore=ignore)]
    if fmt == "int8":  # llm-compressor's own W8A8 INT8 example
        return [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets=["Linear"], scheme="W8A8", ignore=ignore),
        ]
    if fmt == "gptq":
        return [GPTQModifier(targets=["Linear"], scheme="W4A16", ignore=ignore)]
    if fmt == "awq":
        return [AWQModifier(), QuantizationModifier(targets=["Linear"], scheme="W4A16", ignore=ignore)]
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


def make_checkpoint(fmt: str, model_name: str, calib: dict[str, Any]) -> dict[str, Any]:
    """Quantize, then save the compressed checkpoint and the dense rounded weights; report them."""
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fastserve.engine.loader import model_dir
    from fastserve.experiments.m3_library import dense_rounded_state
    from fastserve.quality.text import calibration_ids

    path = model_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    kwargs: dict[str, Any] = {}
    if fmt != "fp8":  # FP8's scales come from the weights alone; the others need calibration data
        ids = calibration_ids(tokenizer, calib["source"], calib["samples"], calib["seq_len"])
        kwargs = {
            "dataset": Dataset.from_dict(
                {"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()}
            ),
            "max_seq_length": calib["seq_len"],
            "num_calibration_samples": calib["samples"],
        }
    start = time.perf_counter()
    oneshot(model=model, recipe=recipe(fmt), **kwargs)
    seconds = time.perf_counter() - start

    name = f"{model_name.split('/')[-1]}-{fmt}"
    out = Path(CHECKPOINT_DIR) / name
    out.mkdir(parents=True, exist_ok=True)
    state, levels = dense_rounded_state(
        model
    )  # before compressing: the save below packs the weights in place
    save_file(state, f"{CHECKPOINT_DIR}/{name}.dense.safetensors")  # not with_suffix: "0.6B" has a dot
    model.save_pretrained(str(out), save_compressed=True)
    tokenizer.save_pretrained(str(out))
    size = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    return {
        "format": fmt,
        "model": model_name,
        "checkpoint": str(out),
        "quantize_s": seconds,
        "checkpoint_bytes": size,
        "max_levels_per_group": max(levels.values()),
    }
