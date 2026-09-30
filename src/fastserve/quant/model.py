"""Applying the quantizers to a whole nanoserve model: calibration hooks, and one entry point per method.

Every function changes the model in place. Embeddings and the LM head stay in 16 bits, as production recipes
do. Qwen3-0.6B ties them together, so quantizing the head would also quantize the embedding.

Calibration inputs flow through the model **layer by layer**: each decoder layer runs on the hidden states the
previous layers produced, and forward hooks capture what each linear layer actually sees.
- **GPTQ** feeds each layer the outputs of the layers *already quantized*, so later layers learn to compensate
  for earlier errors ("sequential"). Within a layer, q/k/v, then o, then gate/up, then down are quantized in
  turn, each measured after the previous ones changed ("true sequential").
- **AWQ** feeds each layer the full-precision outputs, as AutoAWQ does. Its scaling doesn't change the
  function, so only the final rounding differs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from fastserve.engine import rope
from fastserve.quant.awq import AWQConfig, awq_group, awq_quantize_linear
from fastserve.quant.equivalence import input_groups
from fastserve.quant.gptq import GPTQConfig, HessianAccumulator, gptq_quantize, layer_loss
from fastserve.quant.rtn import fake_quantize
from fastserve.quant.w8a8 import QuantLinear, W8A8Config, activation_quantizer, quantize_weight, smooth_group

LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# GPTQ's "true sequential" order: each group is measured after the previous groups were quantized
SUBLAYERS = (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",))


def linear(layer: nn.Module, name: str) -> nn.Linear:
    return getattr(layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp, name)


def decoder_linears(model: nn.Module) -> dict[str, nn.Linear]:
    """Every linear layer inside the decoder layers: {"layers.3.mlp.down_proj": module, ...}."""
    return {
        f"layers.{i}.{'self_attn' if n in LINEAR_NAMES[:4] else 'mlp'}.{n}": linear(layer, n)
        for i, layer in enumerate(model.model.layers)
        for n in LINEAR_NAMES
    }


@contextmanager
def capture_inputs(modules: dict[str, nn.Module], fn: Callable[[str, torch.Tensor], None]) -> Iterator[None]:
    """Call fn(name, input) every time one of the modules runs; the hooks are removed afterwards."""
    handles = [
        module.register_forward_pre_hook(lambda _, args, name=name: fn(name, args[0]))
        for name, module in modules.items()
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class LayerRunner:
    """Runs decoder layers one at a time on calibration hidden states [n, T, d], in batches."""

    def __init__(self, model: nn.Module, ids: torch.Tensor, batch: int):
        cfg = model.config
        n, seq = ids.shape
        self.batch = batch
        positions = torch.arange(seq, device=ids.device).expand(batch, seq)
        self.positions = positions
        dtype = model.model.embed_tokens.weight.dtype
        self.cos, self.sin = rope.cos_sin(positions, cfg.head_dim, cfg.rope_theta, dtype)
        self.hidden = torch.cat([model.model.embed_tokens(chunk) for chunk in ids.split(batch)])  # [n, T, d]

    def run(self, layer: nn.Module) -> torch.Tensor:
        """The layer's outputs on all calibration samples (hooks on the layer fire along the way)."""
        outs = []
        for x in self.hidden.split(self.batch):
            b = x.shape[0]
            outs.append(layer(x, self.cos[:b], self.sin[:b], self.positions[:b], None, None))
        return torch.cat(outs)


# ---- weight-only methods -----------------------------------------------------------------------------------


@torch.no_grad()
def quantize_weights(model: nn.Module, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Replace every decoder linear's weight with fn(weight): RTN, NF4, FP8 weight-only, …"""
    for module in decoder_linears(model).values():
        module.weight.data = fn(module.weight.data)


@torch.no_grad()
def gptq_model(
    model: nn.Module, ids: torch.Tensor, cfg: GPTQConfig, batch: int = 4, true_sequential: bool = True
) -> list[dict[str, Any]]:
    """GPTQ every decoder linear, layer by layer. Returns each linear's objective with RTN and with GPTQ."""
    runner = LayerRunner(model, ids, batch)
    groups = SUBLAYERS if true_sequential else (LINEAR_NAMES,)
    stats = []
    for i, layer in enumerate(model.model.layers):
        for names in groups:
            modules = {name: linear(layer, name) for name in names}
            device = runner.hidden.device
            hessians = {name: HessianAccumulator(m.in_features, device) for name, m in modules.items()}
            with capture_inputs(modules, lambda name, x, acc=hessians: acc[name].add(x)):
                runner.run(layer)
            for name, module in modules.items():
                H, w = hessians[name].H, module.weight.data
                q = gptq_quantize(w, H, cfg)
                stats.append(
                    {
                        "layer": i,
                        "module": name,
                        "rtn_loss": layer_loss(w, fake_quantize(w, cfg.spec), H),
                        "gptq_loss": layer_loss(w, q, H),
                    }
                )
                module.weight.data = q
        runner.hidden = runner.run(layer)  # the quantized layer's outputs feed the next layer
    return stats


@torch.no_grad()
def awq_model(
    model: nn.Module,
    ids: torch.Tensor,
    cfg: AWQConfig,
    batch: int = 4,
    search_tokens: int = 32768,
    clip_tokens: int = 512,
) -> list[dict[str, Any]]:
    """AWQ every decoder layer: search and fold the scales, clip, then round. Returns each group's search."""
    runner = LayerRunner(model, ids, batch)
    stats = []
    for i, layer in enumerate(model.model.layers):
        groups = input_groups(layer)
        # one reader per group sees the group's input; o_proj has no scale group but is still clipped
        readers = {g.name: g.linears[0] for g in groups} | {"o": layer.self_attn.o_proj}
        seen: dict[str, list[torch.Tensor]] = {name: [] for name in readers}
        with capture_inputs(readers, lambda name, x, out=seen: out[name].append(x.reshape(-1, x.shape[-1]))):
            next_hidden = runner.run(layer)  # full-precision outputs feed the next layer (as AutoAWQ)
        inputs = {name: _subsample(torch.cat(xs), search_tokens) for name, xs in seen.items()}
        scales = {}
        for group in groups:
            scales[group.name], result = awq_group(group, inputs[group.name], cfg)
            stats.append({"layer": i, **result})
        attn = layer.self_attn
        for group in groups:
            x = _subsample(inputs[group.name], clip_tokens) / scales[group.name]  # what the scaled layer sees
            for module in group.linears:
                awq_quantize_linear(module, None if module in (attn.q_proj, attn.k_proj) else x, cfg)
        awq_quantize_linear(attn.o_proj, _subsample(inputs["o"], clip_tokens), cfg)
        runner.hidden = next_hidden
    return stats


def _subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    """At most n rows, evenly spaced (deterministic)."""
    if len(x) <= n:
        return x
    return x[torch.linspace(0, len(x) - 1, n, device=x.device).long()]


# ---- weights and activations -------------------------------------------------------------------------------


@torch.no_grad()
def input_amax(model: nn.Module, ids: torch.Tensor, batch: int = 4) -> dict[str, torch.Tensor]:
    """Per-input-channel max |x| of every decoder linear on calibration data: {name: [in]}."""
    amax: dict[str, torch.Tensor] = {}

    def update(name: str, x: torch.Tensor) -> None:
        m = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
        amax[name] = torch.maximum(amax[name], m) if name in amax else m

    with capture_inputs(decoder_linears(model), update):
        for chunk in ids.split(batch):
            model(
                chunk, select=torch.zeros(len(chunk), dtype=torch.long, device=chunk.device)
            )  # skip the head
    return amax


@torch.no_grad()
def smoothquant_model(model: nn.Module, amax: dict[str, torch.Tensor], alpha: float = 0.5) -> dict[str, Any]:
    """Apply SmoothQuant to every scalable input group, using calibration maxima from `input_amax`.

    Updates `amax` in place to the smoothed inputs' maxima (amax / s), for static activation scales.
    """
    names = {id(m): n for n, m in decoder_linears(model).items()}
    for layer in model.model.layers:
        for group in input_groups(layer):
            first = names[id(group.linears[0])]
            s = smooth_group(group, amax[first], alpha)
            for module in group.linears:
                amax[names[id(module)]] = amax[names[id(module)]] / s
    return {"alpha": alpha}


@torch.no_grad()
def w8a8_model(model: nn.Module, cfg: W8A8Config, amax: dict[str, torch.Tensor] | None = None) -> None:
    """Replace every decoder linear with a QuantLinear: 8-bit weights, 8-bit inputs at every call."""
    if cfg.act_static and amax is None:
        raise ValueError("static activation scales need calibration maxima (input_amax)")
    for name, module in decoder_linears(model).items():
        parent_name, child = name.rsplit(".", 1)
        parent = model.model.get_submodule(parent_name)
        static = amax[name].max() if cfg.act_static else None
        quantizer = activation_quantizer(cfg, static)
        setattr(parent, child, QuantLinear(module, quantize_weight(module.weight.data, cfg), quantizer))
