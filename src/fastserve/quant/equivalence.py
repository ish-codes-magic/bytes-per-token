"""Function-preserving rewrites of a Qwen3 decoder layer: move a per-channel scale from a layer's input into
its weights. AWQ and SmoothQuant both work this way.

For a linear layer y = x Wᵀ and any positive s (one number per input channel):

    x Wᵀ = (x / s) (W · s)ᵀ         exact in real arithmetic: nothing about the model changes

The division by s is free: it's folded into whatever produced x. After an RMSNorm, γ ← γ / s. After another
linear layer, that layer's output rows are divided by s. What changes is how hard each side is to quantize.
Scaling a weight column up gives it more of the grid (AWQ). Scaling an activation channel down tames an
outlier (SmoothQuant).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class InputGroup:
    """Linear layers that read the same input, and the module that produces that input."""

    name: str
    prev: nn.Module  # an RMSNorm (divide γ) or an nn.Linear (divide its output rows)
    linears: list[nn.Linear]


def input_groups(layer: nn.Module) -> list[InputGroup]:
    """The scalable inputs of one Qwen3 decoder layer.

    o_proj is left out: its input is attention's output, whose channels come from v_proj, but with GQA v_proj
    has fewer output channels than o_proj has inputs. AutoAWQ skips this pair for the same reason.
    """
    attn, mlp = layer.self_attn, layer.mlp
    return [
        InputGroup("qkv", layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj]),
        InputGroup("gate_up", layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj]),
        # down_proj reads silu(gate) ⊙ up: dividing up's output rows by s divides the product by s
        InputGroup("down", mlp.up_proj, [mlp.down_proj]),
    ]


@torch.no_grad()
def scale_inputs(group: InputGroup, s: torch.Tensor) -> None:
    """Divide the group's input by s [in] (in `prev`) and multiply the readers' weight columns by s."""
    s = s.double()
    if isinstance(group.prev, nn.Linear):
        p = group.prev.weight
        p.data = (p.data.double() / s[:, None]).to(p.dtype)  # output rows
        if group.prev.bias is not None:
            group.prev.bias.data = (group.prev.bias.data.double() / s).to(group.prev.bias.dtype)
    else:
        p = group.prev.weight
        p.data = (p.data.double() / s).to(p.dtype)  # RMSNorm γ
    for linear in group.linears:
        w = linear.weight
        w.data = (w.data.double() * s).to(w.dtype)  # input columns
