"""M3: quantize Qwen3 with every reference method, and measure how far each one moves the model.

Every configuration starts from the same BF16 model, is quantized on a copy, and is compared with BF16 on the
WikiText-2 test windows M2 used: KL divergence per token, top-1 agreement, and perplexity.

Besides that sweep, M3 records what its figures need:
- the outlier atlas (per-channel activation maxima, with and without rotation)
- histograms of normalized weights and activations against each format's grid
- a per-module sensitivity scan
- a GPTQ trace on a small slice of a real layer
- a worked 4-input GPTQ example
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from fastserve.engine.config import ModelConfig
from fastserve.engine.loader import from_state_dict, load_pretrained, model_dir
from fastserve.experiments.m3_library import LIBRARY_DIR
from fastserve.quality.perplexity import evaluate, token_windows
from fastserve.quality.text import calibration_ids, wikitext_eval_ids
from fastserve.quant.awq import AWQConfig
from fastserve.quant.formats import (
    E4M3,
    fp8_fake_quantize,
    nf4_fake_quantize,
    nf4_levels,
    representable_values,
)
from fastserve.quant.gptq import GPTQConfig, HessianAccumulator, gptq_quantize, layer_loss
from fastserve.quant.model import (
    awq_model,
    capture_inputs,
    decoder_linears,
    gptq_model,
    input_amax,
    quantize_weights,
    smoothquant_model,
    w8a8_model,
)
from fastserve.quant.rotation import fold_all_norms, rotate_model
from fastserve.quant.rtn import IntSpec, fake_quantize, grouped
from fastserve.quant.w8a8 import W8A8Config


@dataclass
class Bench:
    """The BF16 reference, its tokenizer, evaluation windows and calibration sets, loaded once per task."""

    model_name: str
    ref: Any
    tokenizer: Any
    windows: torch.Tensor  # [n, window] token ids, on the GPU
    calib: dict[str, Any]  # source, samples, seq_len
    _cache: dict[tuple, torch.Tensor] = field(default_factory=dict)

    @property
    def config(self) -> ModelConfig:
        return self.ref.config

    @property
    def device(self) -> torch.device:
        return self.windows.device

    def calibration(self, source: str | None = None, samples: int | None = None) -> torch.Tensor:
        key = (source or self.calib["source"], samples or self.calib["samples"], self.calib["seq_len"])
        if key not in self._cache:
            self._cache[key] = calibration_ids(self.tokenizer, *key).to(self.device)
        return self._cache[key]


def load_bench(model_name: str, cfg: dict[str, Any]) -> Bench:
    from transformers import AutoTokenizer

    path = model_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    windows = token_windows(wikitext_eval_ids(tokenizer), cfg["eval"]["window"], cfg["eval"]["max_windows"])
    return Bench(model_name, load_pretrained(path), tokenizer, windows.cuda(), cfg["calibration"])


# ---- one configuration --------------------------------------------------------------------------------


def int_spec(entry: dict[str, Any]) -> IntSpec:
    return IntSpec(
        bits=entry["bits"],
        granularity=entry.get("granularity", "group"),
        group_size=entry.get("group_size", 128),
        symmetric=entry.get("symmetric", True),
        full_range=entry.get("full_range", False),
    )


def linear_bits(entry: dict[str, Any], shape: tuple[int, int]) -> float:
    """Storage bits per weight of one decoder linear [out, in] under this configuration (16-bit scales)."""
    method = entry["method"]
    if method in ("rtn", "gptq", "awq", "library"):
        return int_spec(entry).bits_per_weight(shape)
    if method == "nf4":
        return 4 + 16 / entry["block_size"]
    if method in ("fp8_weight", "w8a8"):
        granularity = entry.get("granularity", entry.get("weight_granularity", "channel"))
        return 8 + (16 / shape[1] if granularity == "channel" else 0)
    return 16.0


def size_of(cfg: ModelConfig, entry: dict[str, Any]) -> dict[str, float]:
    """Bits per quantized weight, and the whole model's size with embeddings, LM head and norms in BF16."""
    d, f = cfg.hidden_size, cfg.intermediate_size
    q, kv = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
    shapes = [(q, d), (kv, d), (kv, d), (d, q), (f, d), (f, d), (d, f)]  # q k v o gate up down: [out, in]
    params = sum(o * i for o, i in shapes) * cfg.num_layers
    bits = sum(o * i * linear_bits(entry, (o, i)) for o, i in shapes) * cfg.num_layers
    embeddings = cfg.vocab_size * d * (1 if cfg.tie_word_embeddings and not entry.get("rotate") else 2)
    other = embeddings + d * (2 * cfg.num_layers + 1) + 2 * cfg.head_dim * cfg.num_layers  # + norms
    return {"bits_per_weight": bits / params, "model_gb": (bits + 16 * other) / 8 / 1e9}


def build(bench: Bench, entry: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """A quantized copy of the reference model, and what the method reported along the way."""
    start = time.perf_counter()
    method, info = entry["method"], {}
    if method == "library":
        from safetensors.torch import load_file

        state = load_file(f"{LIBRARY_DIR}/{entry['checkpoint']}.safetensors", device=str(bench.device))
        model = from_state_dict(bench.config, state, device=bench.device, dtype=torch.bfloat16)
    else:
        model = copy.deepcopy(bench.ref)
    if entry.get("dtype") == "float32":  # e.g. to separate rotation's effect from BF16 rounding
        model = model.float()
    if entry.get("rotate") or entry.get("fold"):
        info["gamma_spread"] = gamma_spread(model)
    if entry.get("rotate"):
        rotate_model(model, seed=entry.get("rotation_seed", 0))
    elif entry.get("fold"):
        fold_all_norms(model)
    calib = bench.calibration(entry.get("calibration"), entry.get("samples")) if _needs_data(entry) else None
    if method == "rtn":
        spec = int_spec(entry)
        quantize_weights(model, lambda w: fake_quantize(w, spec))
    elif method == "nf4":
        quantize_weights(model, lambda w: nf4_fake_quantize(w, entry["block_size"]))
    elif method == "fp8_weight":
        quantize_weights(model, lambda w: fp8_fake_quantize(w, E4M3, entry.get("granularity", "channel")))
    elif method == "gptq":
        gcfg = GPTQConfig(
            int_spec(entry),
            block_size=entry.get("block_size", 128),
            damp=entry.get("damp", 0.01),
            act_order=entry.get("act_order", True),
            static_groups=entry.get("static_groups", True),
        )
        stats = gptq_model(model, calib, gcfg, true_sequential=entry.get("true_sequential", False))
        info["objective_rtn"] = sum(s["rtn_loss"] for s in stats)
        info["objective_gptq"] = sum(s["gptq_loss"] for s in stats)
    elif method == "awq":
        acfg = AWQConfig(
            int_spec(entry), duo_scaling=entry.get("duo_scaling", True), clip=entry.get("clip", False)
        )
        stats = awq_model(model, calib, acfg)
        info["alphas"] = [{"layer": s["layer"], "group": s["group"], "alpha": s["alpha"]} for s in stats]
    elif method == "w8a8":
        wcfg = W8A8Config(
            entry["format"],
            entry.get("weight_granularity", "channel"),
            entry["act"],
            entry.get("static", False),
        )
        amax = input_amax(model, calib) if calib is not None else None
        if entry.get("smooth") is not None:
            smoothquant_model(model, amax, entry["smooth"])
        w8a8_model(model, wcfg, amax)
    elif method not in ("bf16", "library"):
        raise ValueError(f"unknown method {method!r}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    info["quantize_s"] = time.perf_counter() - start
    return model, info


def gamma_spread(model: Any) -> dict[str, Any]:
    """How uneven the RMSNorm weights are: folding a large γ into a linear makes an outlier column."""
    norms = {"final": model.model.norm}
    for i, layer in enumerate(model.model.layers):
        norms[f"layers.{i}.input"] = layer.input_layernorm
        norms[f"layers.{i}.post_attention"] = layer.post_attention_layernorm
    spread = {name: (n.weight.abs().max() / n.weight.abs().median()).item() for name, n in norms.items()}
    worst = max(spread, key=spread.get)
    return {
        "worst_norm": worst,
        "max_over_median": spread[worst],
        "typical_max_over_median": sorted(spread.values())[len(spread) // 2],
    }


def _needs_data(entry: dict[str, Any]) -> bool:
    return entry["method"] in ("gptq", "awq") or (
        entry["method"] == "w8a8" and (entry.get("static") or entry.get("smooth") is not None)
    )


def score(bench: Bench, model: Any, windows: torch.Tensor | None = None) -> dict[str, Any]:
    """KL, top-1 agreement and perplexity against the BF16 reference (teacher-forced)."""
    windows = bench.windows if windows is None else windows
    return evaluate(windows, lambda x: bench.ref(x), lambda x: model(x), batch=1)


def run_entry(bench: Bench, entry: dict[str, Any]) -> dict[str, Any]:
    model, info = build(bench, entry)
    result = score(bench, model)
    del model
    torch.cuda.empty_cache()
    return {
        "model": bench.model_name,
        "config": entry["name"],
        "entry": entry,
        **size_of(bench.config, entry),
        **result,
        **info,
    }


# ---- analyses -----------------------------------------------------------------------------------------


def _sig(x: torch.Tensor, digits: int = 4) -> list:
    """A tensor as nested lists of floats with a few significant digits, to keep result files small."""
    return (
        [float(f"{v:.{digits}g}") for v in x.flatten().tolist()]
        if x.dim() <= 1
        else [_sig(r, digits) for r in x]
    )


@torch.no_grad()
def channel_maxima(model: Any, ids: torch.Tensor, batch: int = 4) -> dict[str, torch.Tensor]:
    """Per-channel max |x| of the residual stream (each layer's input and the final norm's) and of down_proj's
    input: {"residual": [layers + 1, d], "down_input": [layers, f]}."""
    layers = model.model.layers
    residual = [None] * (len(layers) + 1)
    down = [None] * len(layers)

    def update(store: list, i: int, x: torch.Tensor) -> None:
        m = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
        store[i] = m if store[i] is None else torch.maximum(store[i], m)

    hooks = [
        layer.register_forward_pre_hook(lambda _, a, i=i: update(residual, i, a[0]))
        for i, layer in enumerate(layers)
    ]
    hooks.append(model.model.norm.register_forward_pre_hook(lambda _, a: update(residual, len(layers), a[0])))
    hooks += [
        layer.mlp.down_proj.register_forward_pre_hook(lambda _, a, i=i: update(down, i, a[0]))
        for i, layer in enumerate(layers)
    ]
    try:
        for chunk in ids.split(batch):
            model(chunk, select=torch.zeros(len(chunk), dtype=torch.long, device=chunk.device))
    finally:
        for hook in hooks:
            hook.remove()
    return {"residual": torch.stack(residual), "down_input": torch.stack(down)}


def outlier_atlas(bench: Bench, samples: int) -> dict[str, Any]:
    ids = bench.calibration(samples=samples)
    plain = channel_maxima(bench.ref, ids)
    rotated_model = copy.deepcopy(bench.ref)
    rotate_model(rotated_model)
    rotated = channel_maxima(rotated_model, ids)
    del rotated_model

    def ratio(m: torch.Tensor) -> float:
        return (m.max() / m.median()).item()

    return {
        "model": bench.model_name,
        "samples": samples,
        "residual": _sig(plain["residual"], 3),
        "down_input": _sig(plain["down_input"], 3),
        "residual_rotated": _sig(rotated["residual"], 3),
        "residual_ratio": ratio(plain["residual"]),
        "residual_rotated_ratio": ratio(rotated["residual"]),
        "down_input_ratio": ratio(plain["down_input"]),
        "top_channels": plain["residual"].amax(dim=0).topk(5).indices.tolist(),
    }


def histograms(bench: Bench, layer: int, bins: int = 200, group_size: int = 128) -> dict[str, Any]:
    """Weights of one layer divided by their group's max, activations divided by their token's max, and the
    grid points each format offers on that same normalized scale."""
    edges = torch.linspace(-1, 1, bins + 1)
    weights = torch.cat(
        [
            (g / g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)).flatten()
            for name, m in decoder_linears(bench.ref).items()
            if name.startswith(f"layers.{layer}.")
            for g in [grouped(m.weight.float(), IntSpec(4, "group", group_size))]
        ]
    ).cpu()
    seen: list[torch.Tensor] = []
    q_proj = bench.ref.model.layers[layer].self_attn.q_proj
    with (
        capture_inputs({"q": q_proj}, lambda _, x: seen.append(x.reshape(-1, x.shape[-1]).float())),
        torch.no_grad(),
    ):
        bench.ref(
            bench.calibration(samples=4)[:2], select=torch.zeros(2, dtype=torch.long, device=bench.device)
        )
    acts = torch.cat(seen)
    acts = (acts / acts.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)).flatten().cpu()
    fp8 = representable_values(E4M3) / E4M3.max_value
    return {
        "model": bench.model_name,
        "layer": layer,
        "edges": _sig(edges),
        "weights": torch.histc(weights, bins, -1, 1).long().tolist(),
        "activations": torch.histc(acts, bins, -1, 1).long().tolist(),
        "activation_abs_max_over_median": (acts.abs().max() / acts.abs().median()).item(),
        "grids": {
            "INT4 (textbook)": [i / 7 for i in range(-7, 8)],
            "INT4 (full range)": [i / 7.5 for i in range(-8, 8)],
            "NF4": nf4_levels(),
            "FP8 E4M3": _sig(fp8),
        },
    }


@torch.no_grad()
def _reference_logprobs(bench: Bench, windows: torch.Tensor) -> list[torch.Tensor]:
    return [torch.log_softmax(bench.ref(w[None])[0, :-1].float(), dim=-1) for w in windows]


@torch.no_grad()
def _kl(model: Any, windows: torch.Tensor, ref_logprobs: list[torch.Tensor]) -> float:
    total, n = 0.0, 0
    for w, ref in zip(windows, ref_logprobs, strict=True):
        cand = torch.log_softmax(model(w[None])[0, :-1].float(), dim=-1)
        total += (ref.exp() * (ref - cand)).sum().item()
        n += len(ref)
    return total / n


@torch.no_grad()
def sensitivity(bench: Bench, entry: dict[str, Any]) -> dict[str, Any]:
    """KL when only one decoder linear is quantized (RTN), for every layer and module type."""
    spec, windows = int_spec(entry), bench.windows[: entry["windows"]]
    ref = _reference_logprobs(bench, windows)
    model = bench.ref
    cells = []
    for name, module in decoder_linears(model).items():
        original = module.weight.data
        module.weight.data = fake_quantize(original, spec)
        try:
            kl = _kl(model, windows, ref)
        finally:
            module.weight.data = original
        _, layer, _, kind = name.split(".")
        cells.append({"layer": int(layer), "module": kind, "kl": kl})
    return {"model": bench.model_name, "spec": spec.label, "windows": len(windows), "cells": cells}


@torch.no_grad()
def gptq_trace(bench: Bench, entry: dict[str, Any]) -> dict[str, Any]:
    """GPTQ on a small slice of a real layer, recording every column: the data behind the animation."""
    layer = bench.ref.model.layers[entry["layer"]]
    module = getattr(layer.self_attn, entry["module"])
    rows, cols = entry["rows"], entry["cols"]
    acc = HessianAccumulator(module.in_features, bench.device)
    with capture_inputs({"m": module}, lambda _, x: acc.add(x)):
        bench.ref(bench.calibration(samples=8), select=torch.zeros(8, dtype=torch.long, device=bench.device))
    w, H = module.weight.data[:rows, :cols].float(), acc.H[:cols, :cols]
    spec = IntSpec(entry["bits"], "channel")
    record: list = []
    q = gptq_quantize(w, H, GPTQConfig(spec, block_size=cols), record=record)
    rtn = fake_quantize(w, spec)
    return {
        "model": bench.model_name,
        "layer": entry["layer"],
        "module": entry["module"],
        "spec": spec.label,
        "weights": _sig(w),
        "rtn": _sig(rtn),
        "frames": [{"column": r["column"], "weights": _sig(r["weights"])} for r in record],
        "loss_rtn": layer_loss(w, rtn, H),
        "loss_gptq": layer_loss(w, q, H),
    }


def worked_example() -> dict[str, Any]:
    """GPTQ on one row of 4 weights, with correlated inputs, every step recorded (for the learning doc).

    Inputs 0 and 1 almost always move together, and 2 and 3 move in opposite directions. The weights are
    chosen so the story is visible: RTN rounds both 0.55s down, and because their inputs move together the two
    errors add up; GPTQ rounds the first down, pushes the second up, and the errors cancel.
    """
    w = torch.tensor([[0.55, 0.55, -0.4, 1.2]], dtype=torch.float64)
    x = torch.tensor(  # [samples, 4]
        [
            [1.0, 0.95, 0.3, -0.2],
            [0.7, 0.8, -0.5, 0.6],
            [-0.6, -0.5, 1.0, -0.9],
            [0.2, 0.3, 0.7, -0.8],
            [-1.0, -0.9, -0.2, 0.3],
            [0.5, 0.4, -0.9, 1.0],
        ],
        dtype=torch.float64,
    )
    H = 2 * x.T @ x / len(x)
    spec = IntSpec(3, "channel")
    record: list = []
    cfg = GPTQConfig(spec, block_size=4, damp=0.0)
    q = gptq_quantize(w.float(), H.float(), cfg, record=record)
    rtn = fake_quantize(w.float(), spec)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    return {
        "w": w[0].tolist(),
        "x": x.tolist(),
        "H": H.tolist(),
        "U": U.tolist(),
        "spec": spec.label,
        "steps": [
            {"column": r["column"], "error": r["error"].tolist(), "weights": r["weights"][0].tolist()}
            for r in record
        ],
        "gptq": q[0].tolist(),
        "rtn": rtn[0].tolist(),
        "loss_gptq": layer_loss(w.float(), q, H.float()),
        "loss_rtn": layer_loss(w.float(), rtn, H.float()),
    }


# ---- tasks --------------------------------------------------------------------------------------------


def run_task(
    task: str, config: dict[str, Any], *, run_id: str, git: dict | None, config_path: str
) -> list[dict]:
    """Run one section of benchmarks/configs/m3_quant.yaml in this container; one record per result."""
    from fastserve.results import environment_info, make_record

    spec = config["tasks"][task]
    model_name = spec.get("model", config["model"]) if isinstance(spec, dict) else config["model"]
    meta = {"path": config_path, "task": task, "eval": config["eval"], "calibration": config["calibration"]}
    env = environment_info()

    def record(experiment: str, metrics: dict[str, Any]) -> dict:
        return make_record(experiment, metrics, run_id=run_id, config=meta, git=git, env=env)

    bench = load_bench(model_name, config)
    if task == "analyses":
        return [
            record("m3_outliers", outlier_atlas(bench, spec["outliers"]["samples"])),
            record("m3_histograms", histograms(bench, spec["histograms"]["layer"])),
            record("m3_gptq_trace", gptq_trace(bench, spec["trace"])),
            record("m3_worked_example", worked_example()),
        ]
    if task == "sensitivity":
        return [record("m3_sensitivity", sensitivity(bench, spec))]
    records = []
    for entry in spec["entries"] if isinstance(spec, dict) else spec:
        start = time.perf_counter()
        metrics = run_entry(bench, entry)
        print(
            f"{model_name} {entry['name']}: KL {metrics['mean_kl']:.4g}, "
            f"perplexity {metrics['perplexity_cand']:.3f} ({time.perf_counter() - start:.0f} s)",
            flush=True,
        )
        records.append(record("m3_config", metrics))
    return records
