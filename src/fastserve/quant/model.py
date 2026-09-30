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
from dataclasses import dataclass
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
    search_samples: int = 16,
    clip_tokens: int = 512,
) -> list[dict[str, Any]]:
    """AWQ every decoder layer: search and fold the scales, optionally clip, then round.

    Channel statistics x̄ come from every calibration token. The α search runs each group's parent module (the
    attention block, the MLP, or down_proj itself) on the first `search_samples` sequences. Returns each
    group's search.
    """
    runner = LayerRunner(model, ids, batch)
    stats = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        seen = _awq_capture(layer, runner, search_samples, clip_tokens)
        scales = {}
        for group in input_groups(layer):
            calls = seen.calls[group.name]
            parent = seen.parents[group.name]
            scales[group.name], result = awq_group(
                group, seen.x_mean[group.name], lambda p=parent, c=calls: torch.cat([p(*a) for a in c]), cfg
            )
            stats.append({"layer": i, **result})
        for group in input_groups(layer):
            x = seen.clip_inputs[group.name] / scales[group.name]  # what the scaled layer will see
            for module in group.linears:  # q and k are not clipped: softmax amplifies their errors
                awq_quantize_linear(module, None if module in (attn.q_proj, attn.k_proj) else x, cfg)
        awq_quantize_linear(attn.o_proj, seen.clip_inputs["o"], cfg)
        runner.hidden = seen.next_hidden
    return stats


@dataclass
class _AWQInputs:
    parents: dict[str, nn.Module]  # the module whose output each group's search preserves
    calls: dict[str, list[tuple]]  # the parent's arguments on the first sequences
    x_mean: dict[str, torch.Tensor]  # mean |x| per input channel, over every calibration token
    clip_inputs: dict[str, torch.Tensor]  # a few hundred input rows per group, for the clip search
    next_hidden: torch.Tensor  # the layer's full-precision outputs


def _awq_capture(layer: nn.Module, runner: LayerRunner, search_samples: int, clip_tokens: int) -> _AWQInputs:
    """Run one layer on all calibration data, recording what AWQ's searches need."""
    attn, mlp = layer.self_attn, layer.mlp
    groups = input_groups(layer)
    parents = {"qkv": attn, "gate_up": mlp, "down": mlp.down_proj}
    calls: dict[str, list[tuple]] = {name: [] for name in parents}
    sums = {g.name: torch.zeros(g.linears[0].in_features, device=runner.hidden.device) for g in groups}
    counts = dict.fromkeys(sums, 0)
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in (*sums, "o")}

    def keep_args(name: str, args: tuple) -> None:
        if sum(a[0].shape[0] for a in calls[name]) < search_samples:
            calls[name].append(args)

    def observe(name: str, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        if name in sums:
            sums[name] += flat.abs().float().sum(dim=0)
            counts[name] += len(flat)
        rows[name].append(_subsample(flat, clip_tokens))

    handles = [p.register_forward_pre_hook(lambda _, a, n=n: keep_args(n, a)) for n, p in parents.items()]
    readers = {g.name: g.linears[0] for g in groups} | {"o": attn.o_proj}
    try:
        with capture_inputs(readers, observe):
            next_hidden = runner.run(layer)  # full-precision outputs feed the next layer (as AutoAWQ)
    finally:
        for handle in handles:
            handle.remove()
    return _AWQInputs(
        parents=parents,
        calls=calls,
        x_mean={name: sums[name] / counts[name] for name in sums},
        clip_inputs={name: _subsample(torch.cat(r), clip_tokens) for name, r in rows.items()},
        next_hidden=next_hidden,
    )


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
def w8a8_model(
    model: nn.Module,
    cfg: W8A8Config,
    amax: dict[str, torch.Tensor] | None = None,
    quantize_weights: bool = True,
) -> None:
    """Replace every decoder linear with a QuantLinear: 8-bit weights, 8-bit inputs at every call.

    quantize_weights=False keeps the weights as they are, for checkpoints whose weights were already rounded
    (by a library, on its own grid) and only need their activations quantized.
    """
    if cfg.act_static and amax is None:
        raise ValueError("static activation scales need calibration maxima (input_amax)")
    for name, module in decoder_linears(model).items():
        parent_name, child = name.rsplit(".", 1)
        parent = model.model.get_submodule(parent_name)
        static = amax[name].max() if cfg.act_static else None
        quantizer = activation_quantizer(cfg, static)
        weight = quantize_weight(module.weight.data, cfg) if quantize_weights else module.weight.data
        setattr(parent, child, QuantLinear(module, weight, quantizer))
