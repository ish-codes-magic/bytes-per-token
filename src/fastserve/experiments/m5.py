"""M5, the reference side: KV-cache quantization and eviction in nanoserve, judged before any kernel exists.

Each task runs in its own container (benchmarks/configs/m5_kv.yaml, `tasks`):
- kl:     every KV policy against the BF16 cache, teacher-forced on M2's WikiText-2 windows: KL per token,
          top-1 agreement, perplexity. The weights stay BF16, so only the cache changes.
- needle: M2's needle-in-a-haystack grid through nanoserve with each policy: long prompts prefilled in chunks
          through a real cache, then a greedy answer. Recall at long context is what quantization and
          eviction put at risk, and what KL on 2k-token windows can't see.
- stats:  per-channel magnitudes of the cached keys and values: why keys are harder to quantize.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from fastserve.engine.generate import generate_long
from fastserve.engine.model import use_fused_attention
from fastserve.engine.sampler import SamplingParams
from fastserve.experiments.m3 import Bench, load_bench
from fastserve.kv.quant import apply_kv_policy
from fastserve.kv.sizing import KVSpec, TensorQuant, bytes_per_token
from fastserve.quality.needle import grid, passed
from fastserve.quality.perplexity import evaluate


def kv_spec(entry: dict[str, Any]) -> KVSpec:
    """A KVSpec from a config entry, e.g. {name: int4-kivi, keys: {bits: 4, axis: channel, group: 32}}."""

    def tensor(cfg: dict[str, Any] | None) -> TensorQuant | None:
        return None if cfg is None else TensorQuant(**cfg)

    return KVSpec(
        name=entry["name"],
        keys=tensor(entry.get("keys")),
        values=tensor(entry.get("values")),
        rotate_keys=entry.get("rotate_keys", False),
        sinks=entry.get("sinks"),
        window=entry.get("window"),
    )


def _sizes(bench_or_cfg: Any, spec: KVSpec) -> dict[str, float]:
    cfg = bench_or_cfg.config if isinstance(bench_or_cfg, Bench) else bench_or_cfg
    return {"bits_per_element": spec.bits_per_element(), "bytes_per_token": bytes_per_token(cfg, spec)}


@torch.no_grad()
def kv_kl(bench: Bench, entry: dict[str, Any]) -> dict[str, Any]:
    """One policy against the BF16 cache on the same model: the reference runs with no policy."""
    spec, model = kv_spec(entry), bench.ref

    def reference(x: torch.Tensor) -> torch.Tensor:
        apply_kv_policy(model, None)
        return model(x)

    def candidate(x: torch.Tensor) -> torch.Tensor:
        apply_kv_policy(model, spec)
        return model(x)

    try:
        metrics = evaluate(bench.windows, reference, candidate, batch=1)
    finally:
        apply_kv_policy(model, None)
    return {"model": bench.model_name, "policy": spec.name, **_sizes(bench, spec), **metrics}


@torch.no_grad()
def kv_needle(bench: Bench, entry: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """M2's needle grid through nanoserve with one policy. A cell passes when the answer has the secret."""
    spec, model, tokenizer = kv_spec(entry), bench.ref, bench.tokenizer
    stop = tuple(tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
    params = SamplingParams(max_new_tokens=cfg["max_new_tokens"], stop_token_ids=stop)
    apply_kv_policy(model, spec)
    cells = []
    try:
        for case in grid(tokenizer, cfg["lengths"], cfg["depths"], cfg["secrets"]):
            prompt = tokenizer.encode(case.prompt, add_special_tokens=False)
            answer = tokenizer.decode(
                generate_long(model, prompt, params, chunk=cfg["chunk"]), skip_special_tokens=True
            )
            cells.append(
                {
                    "length": case.context_tokens,
                    "depth": case.depth,
                    "secret": case.secret,
                    "passed": passed(case, answer),
                    "answer": answer.strip()[:80],
                }
            )
    finally:
        apply_kv_policy(model, None)
    rate = sum(c["passed"] for c in cells) / len(cells)
    return {"model": bench.model_name, "policy": spec.name, "cells": cells, "pass_rate": rate}


class _Recorder:
    """Stands in for a KV policy to watch what the cache would store: per-channel max |value| per head."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None  # [kv_heads, head_dim]
        self.v: torch.Tensor | None = None

    def store(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_max, v_max = (
            k.abs().amax(dim=(0, 2)).float(),
            v.abs().amax(dim=(0, 2)).float(),
        )  # over batch, tokens
        self.k = k_max if self.k is None else torch.maximum(self.k, k_max)
        self.v = v_max if self.v is None else torch.maximum(self.v, v_max)
        return k, v

    def visible(self, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return mask


def _ratio(channel_max: torch.Tensor) -> float:
    """Largest channel ÷ median channel, per head, then the median over heads."""
    return (channel_max.amax(-1) / channel_max.median(-1).values).median().item()


@torch.no_grad()
def kv_stats(bench: Bench, windows: int, layers: list[int]) -> dict[str, Any]:
    """How outlier-heavy the cached keys and values are, per layer; full channel profiles for a few layers."""
    model = bench.ref
    recorders = [_Recorder() for _ in model.model.layers]
    for layer, rec in zip(model.model.layers, recorders, strict=True):
        layer.self_attn.kv_policy = rec
    try:
        for w in bench.windows[:windows]:
            model(w[None])
    finally:
        apply_kv_policy(model, None)

    def sig(x: torch.Tensor) -> list:
        return [[float(f"{v:.4g}") for v in row] for row in x.tolist()]

    return {
        "model": bench.model_name,
        "windows": windows,
        "key_ratio": [_ratio(r.k) for r in recorders],  # per layer
        "value_ratio": [_ratio(r.v) for r in recorders],
        "profiles": {str(i): {"keys": sig(recorders[i].k), "values": sig(recorders[i].v)} for i in layers},
    }


def run_task(
    task: str, config: dict[str, Any], *, run_id: str, git: dict | None, config_path: str
) -> list[dict]:
    """Run one section of benchmarks/configs/m5_kv.yaml's `tasks` in this container; one record per result."""
    from fastserve.results import environment_info, make_record

    spec = config["tasks"][task]
    meta = {"path": config_path, "task": task, "eval": config["eval"]}
    env = environment_info()

    def record(experiment: str, metrics: dict[str, Any]) -> dict:
        return make_record(experiment, metrics, run_id=run_id, config=meta, git=git, env=env)

    bench = load_bench(spec["model"], config)
    use_fused_attention(bench.ref)  # tested against the reference attention; needed for 32k prompts
    policies = [
        p
        for p in config["policies"]
        if p["name"] in spec.get("policies", [p["name"] for p in config["policies"]])
    ]
    if spec["kind"] == "stats":
        return [record("m5_kv_stats", kv_stats(bench, spec["windows"], spec["layers"]))]
    records = []
    for entry in policies:
        start = time.perf_counter()
        if spec["kind"] == "kl":
            metrics = kv_kl(bench, entry)
            note = f"KL {metrics['mean_kl']:.4g}"
            records.append(record("m5_kv_kl", metrics))
        else:
            metrics = kv_needle(bench, entry, config["needle"])
            note = f"needle pass rate {metrics['pass_rate']:.2f}"
            records.append(record("m5_kv_needle", metrics))
        print(f"{spec['model']} {entry['name']}: {note} ({time.perf_counter() - start:.0f} s)", flush=True)
    return records
