"""Read llm-compressor's compressed-tensors checkpoints back into dense BF16 weights, in plain PyTorch.

M4's checkpoints store each quantized Linear in one of three formats (`quantization_config.format` in
config.json), verified against compressed-tensors 0.19 and the checkpoints themselves:

- **float-quantized** (FP8 W8A8): `weight` float8_e4m3fn [out, in] and `weight_scale` [out, 1], an FP8
  value per weight and a scale per row.
- **int-quantized** (INT8 W8A8): `weight` int8 [out, in] and `weight_scale` [out, 1].
- **pack-quantized** (INT4 W4A16): `weight_packed` int32 [out, in·4/32] holding 8 values per word,
  `weight_scale` [out, in/128] (a scale per group of 128) and `weight_shape` (the unpacked shape).

Every grid is symmetric (no zero-point), so dequantizing is always `value × scale of its row and group`. With
this, a published checkpoint is the single source of truth: nanoserve rebuilds the exact rounded weights from
it, with no separately stored dense copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def pack_int(q: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Pack signed integers [rows, cols] into int32 words [rows, cols·bits/32], as compressed-tensors does.

    Each value is stored unsigned, as q + 2^(bits−1). Element i of a row sits at bits i·b … i·b+b−1, counting
    from the least significant bit of the first word. A row that doesn't fill its last word is padded with
    zero bits, which `unpack_int` cuts off again. Only widths that divide 32 (2, 4, 8) are supported: no value
    straddles two words.
    """
    if 32 % num_bits:
        raise ValueError(f"only bit widths that divide 32 are supported, got {num_bits}")
    per_word = 32 // num_bits
    rows, cols = q.shape
    unsigned = q.to(torch.int64) + (1 << (num_bits - 1))
    unsigned = torch.nn.functional.pad(unsigned, (0, -cols % per_word)).reshape(rows, -1, per_word)
    shifts = torch.arange(per_word, dtype=torch.int64) * num_bits  # [per_word]
    words = (unsigned << shifts).sum(dim=-1)  # [rows, words]: the fields don't overlap, so sum = OR
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)  # reinterpret as signed int32


def unpack_int(packed: torch.Tensor, cols: int, num_bits: int = 4) -> torch.Tensor:
    """Inverse of `pack_int`: int32 words [rows, words] → signed int8 values [rows, cols]."""
    if 32 % num_bits:
        raise ValueError(f"only bit widths that divide 32 are supported, got {num_bits}")
    per_word = 32 // num_bits
    shifts = torch.arange(per_word, dtype=torch.int32, device=packed.device) * num_bits  # [per_word]
    # An arithmetic right shift drags the sign bit along, but the mask keeps only the field's own bits.
    fields = (packed.unsqueeze(-1) >> shifts) & ((1 << num_bits) - 1)  # [rows, words, per_word]
    values = fields.reshape(packed.shape[0], -1)[:, :cols]
    return (values - (1 << (num_bits - 1))).to(torch.int8)


def dequantize_linear(tensors: dict[str, torch.Tensor], num_bits: int = 4) -> torch.Tensor:
    """One Linear's dense BF16 weight [out, in] from its stored tensors (keys without the module prefix)."""
    if "weight_zero_point" in tensors:
        raise NotImplementedError("asymmetric checkpoints (with a zero-point) aren't used in this project")
    scale = tensors["weight_scale"].float()  # [out, groups]
    if "weight_packed" in tensors:
        out, cols = (int(x) for x in tensors["weight_shape"])
        values = unpack_int(tensors["weight_packed"], cols, num_bits).float()  # [out, in]
    else:
        values = tensors["weight"].float()  # FP8 or INT8, [out, in]: exact in float32
        cols = values.shape[1]
    group = cols // scale.shape[1]  # in/groups: the whole row for per-channel scales
    # Each product of a ≤8-bit value and a BF16 scale is exact in float32, so one rounding to BF16 follows.
    return (values * scale.repeat_interleave(group, dim=1)).to(torch.bfloat16)


def dense_state_dict(state: dict[str, torch.Tensor], quantization_config: dict) -> dict[str, torch.Tensor]:
    """A compressed state dict with every quantized Linear replaced by its dense BF16 `weight`.

    Everything else (embedding, norms, the unquantized LM head) passes through unchanged. Runtime-only
    tensors such as activation scales are dropped: nanoserve quantizes activations itself when asked.
    """
    bits = {g["weights"]["num_bits"] for g in quantization_config["config_groups"].values()}
    if len(bits) != 1:
        raise NotImplementedError(f"one weight bit width per checkpoint expected, got {sorted(bits)}")
    (num_bits,) = bits
    modules = {key.rsplit(".", 1)[0] for key in state if key.endswith(".weight_scale")}
    dense = {}
    for key, tensor in state.items():
        module, name = key.rsplit(".", 1)
        if module not in modules:
            dense[key] = tensor
        elif name in ("weight", "weight_packed"):
            parts = {
                n: state[f"{module}.{n}"]
                for n in ("weight", "weight_packed", "weight_scale", "weight_shape")
                if f"{module}.{n}" in state
            }
            dense[f"{module}.weight"] = dequantize_linear(parts, num_bits)
        elif name == "bias":
            dense[key] = tensor
    return dense


def load_dense(folder: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read a compressed-tensors checkpoint folder (config.json + *.safetensors) into a dense state dict."""
    from safetensors.torch import load_file

    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    state: dict[str, torch.Tensor] = {}
    for file in sorted(folder.glob("*.safetensors")):
        state.update(load_file(file, device=device))
    return dense_state_dict(state, config["quantization_config"])
