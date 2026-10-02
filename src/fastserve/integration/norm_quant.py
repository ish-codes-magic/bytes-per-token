"""Kernel 1 inside nanoserve: the two pre-norms of every layer emit INT8-quantized activations.

In a W8A8-INT8 model the linear layers after each RMSNorm (q/k/v, and gate/up) read 8-bit activations.
`NormQuantInt8` makes nanoserve's norms produce exactly that, through the reference (norm, then quantize) or
through the fused kernel, so the two can be compared inside the real model: same logits, one pass over
memory instead of two.

nanoserve has no INT8 matmul, so the codes are multiplied back by their scale before the next layer reads
them. That last step exists only here; in vLLM the codes and the scale go straight into `cutlass_scaled_mm`.
"""

from __future__ import annotations

import torch
from torch import nn

from fastserve.kernels import reference


class NormQuantInt8(nn.Module):
    """An RMSNorm whose output lies on a per-token INT8 grid."""

    def __init__(self, norm: nn.Module, fused: bool):
        super().__init__()
        self.weight, self.eps, self.fused = norm.weight, norm.eps, fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            from fastserve.kernels.norm_quant import rms_norm_int8

            codes, scale = rms_norm_int8(x, self.weight, self.eps)
        else:
            codes, scale = reference.rms_norm_int8(x, self.weight, self.eps)
        return reference.dequantize_int8(codes, scale).to(x.dtype)


def quantize_norm_outputs(model: nn.Module, fused: bool) -> nn.Module:
    """Swap both pre-norms of every decoder layer for `NormQuantInt8`. Changes the model in place."""
    for layer in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            setattr(layer, name, NormQuantInt8(getattr(layer, name), fused))  # also re-wraps a wrapped norm
    return model
