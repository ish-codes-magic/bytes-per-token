"""M1 measurements: nanoserve running the real Qwen3-0.6B on a GPU.

hf_parity            nanoserve vs Hugging Face logits on fixed prompts (BF16)
decode_speed         time per decode step at several batch sizes
prefill_speed        time to process a prompt, at several lengths
kernel_profile       torch.profiler: GPU kernels per step, and how much of the step the GPU is busy
component_times      where a step's time goes: embedding, norms, attention, MLP, LM head
attention_maps       attention probabilities for a few layers, and "attention sink" strength per layer
continuous_batching  a ContinuousBatcher run whose per-step log feeds the block-table animation
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

import fastserve.engine.model as model_module
from fastserve.engine.attention import repeat_kv
from fastserve.engine.generate import ContinuousBatcher, Request
from fastserve.engine.kv_cache import ContiguousKVCache, PagedKVCache
from fastserve.engine.loader import load_pretrained
from fastserve.engine.model import CausalLM
from fastserve.engine.sampler import SamplingParams
from fastserve.results import environment_info, make_record, new_run_id
from fastserve.timing import cuda_time_ms, summarize

WARMUP = 5


def _random_ids(vocab: int, shape: tuple[int, ...], seed: int, device: str = "cuda") -> torch.Tensor:
    return torch.randint(0, vocab, shape, generator=torch.Generator().manual_seed(seed)).to(device)


# -- correctness -----------------------------------------------------------------------------------


@torch.inference_mode()
def hf_parity(
    model_dir: str, ours: CausalLM, prompts: list[str], *, attn_implementation: str = "eager"
) -> list[dict[str, Any]]:
    """Per prompt: how closely nanoserve's logits track Hugging Face's (same weights, same dtype).

    "eager" is Hugging Face's plain-PyTorch attention, the same math in the same order as ours. "sdpa" uses
    PyTorch's fused attention kernel: comparing against it is a negative control, showing the comparison can
    detect small numerical differences at all.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dtype = ours.lm_head.weight.dtype
    hf = (
        AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn_implementation)
        .cuda()
        .eval()
    )
    rows = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        expected, got = hf(ids).logits.float()[0], ours(ids).float()[0]  # [tokens, vocab]
        diff = (expected - got).abs()
        kl = F.kl_div(got.log_softmax(-1), expected.log_softmax(-1), log_target=True, reduction="none").sum(
            -1
        )
        rows.append(
            {
                "hf_attention": attn_implementation,
                "prompt": prompt,
                "tokens": ids.shape[1],
                "max_abs_diff": diff.max().item(),
                "mean_abs_diff": diff.mean().item(),
                "max_abs_logit": expected.abs().max().item(),
                "top1_agreement": (expected.argmax(-1) == got.argmax(-1)).float().mean().item(),
                "mean_kl_hf_to_ours": kl.mean().item(),
            }
        )
    del hf
    torch.cuda.empty_cache()
    return rows


# -- speed -----------------------------------------------------------------------------------------


@torch.inference_mode()
def decode_speed(model: CausalLM, *, batch: int, prompt_len: int, steps: int, seed: int) -> dict[str, Any]:
    """Prefill `batch` random prompts (untimed), then time greedy decode steps, argmax included."""
    cfg, weight = model.config, model.lm_head.weight
    cache = ContiguousKVCache(
        cfg, max_batch=batch, max_len=prompt_len + WARMUP + steps, dtype=weight.dtype, device=weight.device
    )
    prompts = _random_ids(cfg.vocab_size, (batch, prompt_len), seed)
    positions = torch.arange(prompt_len, device="cuda").expand(batch, -1)
    last = torch.full((batch,), prompt_len - 1, device="cuda")
    tokens = model(prompts, positions, cache, select=last).argmax(-1, keepdim=True)  # [B, 1]
    position = torch.full((batch, 1), prompt_len, device="cuda")
    zeros = torch.zeros(batch, dtype=torch.long, device="cuda")

    def step() -> None:
        nonlocal tokens, position
        tokens = model(tokens, position, cache, select=zeros).argmax(-1, keepdim=True)
        position = position + 1

    stats = summarize(cuda_time_ms(step, warmup=WARMUP, iters=steps))
    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "steps": steps,
        "step_ms_median": stats.median_ms,
        "tokens_per_s": batch / (stats.median_ms / 1e3),
        "timing": stats.to_dict(),
    }


@torch.inference_mode()
def prefill_speed(model: CausalLM, *, length: int, repeats: int, seed: int) -> dict[str, Any]:
    """Time one forward pass over a `length`-token prompt (batch 1), writing the KV cache."""
    cfg, weight = model.config, model.lm_head.weight
    cache = ContiguousKVCache(cfg, max_batch=1, max_len=length, dtype=weight.dtype, device=weight.device)
    ids = _random_ids(cfg.vocab_size, (1, length), seed)
    positions = torch.arange(length, device="cuda")[None]
    last = torch.tensor([length - 1], device="cuda")
    stats = summarize(
        cuda_time_ms(lambda: model(ids, positions, cache, select=last), warmup=2, iters=repeats)
    )
    return {"length": length, "ms_median": stats.median_ms, "timing": stats.to_dict()}


# -- where the time goes ---------------------------------------------------------------------------

_KERNEL_CATEGORIES = [
    ("matmul", re.compile(r"gemm|gemv|cutlass|xmma|cublas|splitk", re.I)),
    ("softmax", re.compile(r"softmax", re.I)),
    ("reduction", re.compile(r"reduce", re.I)),
    ("copy / index / cat", re.compile(r"copy|index|gather|scatter|cat|memcpy|memset", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled", re.I)),
]


def _category(kernel_name: str) -> str:
    return next((name for name, pattern in _KERNEL_CATEGORIES if pattern.search(kernel_name)), "other")


def kernel_profile(fn, *, label: str) -> dict[str, Any]:
    """Count the GPU kernels one call of `fn` launches, and sum their GPU time, by kernel category."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    by_category: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "ms": 0.0})
    for k in kernels:
        bucket = by_category[_category(k.name)]
        bucket["count"] += 1
        bucket["ms"] += k.time_range.elapsed_us() / 1e3
    return {
        "label": label,
        "kernels": len(kernels),
        "kernel_ms": sum(b["ms"] for b in by_category.values()),
        "by_category": dict(by_category),
    }


@contextmanager
def _component_events(model: CausalLM):
    """Record CUDA events around the model's main pieces. Hooks are removed afterwards (no lasting cost)."""
    events: dict[str, list[list[torch.cuda.Event]]] = defaultdict(list)
    handles = []

    def watch(module: torch.nn.Module, name: str) -> None:
        def before(*_):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            events[name].append([start])

        def after(*_):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events[name][-1].append(end)

        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])

    watch(model.model.embed_tokens, "embedding")
    for layer in model.model.layers:
        watch(layer.input_layernorm, "RMSNorm")
        watch(layer.self_attn, "attention block")
        watch(layer.post_attention_layernorm, "RMSNorm")
        watch(layer.mlp, "MLP")
    watch(model.model.norm, "RMSNorm")
    watch(model.lm_head, "LM head")
    try:
        yield events
    finally:
        for handle in handles:
            handle.remove()


def component_times(model: CausalLM, fn, *, label: str) -> dict[str, Any]:
    """Split one call of `fn` into model components (GPU-timeline time, launch gaps included)."""
    fn()
    with _component_events(model) as events:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
    parts = {name: sum(s.elapsed_time(e) for s, e in pairs) for name, pairs in events.items()}
    total = start.elapsed_time(end)
    parts["other (RoPE tables, sampling, Python)"] = max(total - sum(parts.values()), 0.0)
    return {"label": label, "total_ms": total, "components_ms": parts}


# -- what attention looks at -----------------------------------------------------------------------


@torch.inference_mode()
def attention_maps(model: CausalLM, ids: torch.Tensor, layers: list[int], head: int) -> dict[str, Any]:
    """Attention probabilities of one head in selected layers, plus the average weight each layer puts on the
    first token (the "attention sink") across all heads and later query positions."""
    captured: list[torch.Tensor] = []
    original = model_module.attention

    def recording(q, k, v, mask, scale):
        scores = (q @ repeat_kv(k, q.shape[1] // k.shape[1]).transpose(-2, -1)) * scale
        captured.append(torch.softmax(scores.masked_fill(~mask, float("-inf")).float(), -1)[0].cpu())
        return original(q, k, v, mask, scale)

    model_module.attention = recording  # layers call attention() in order, so captured[i] is layer i
    try:
        model(ids)
    finally:
        model_module.attention = original
    sink = [probs[:, 1:, 0].mean().item() for probs in captured]  # [heads, T, T]: queries after the first
    return {
        "tokens": ids.shape[1],
        "head": head,
        "maps": {str(layer): captured[layer][head].tolist() for layer in layers},
        "sink_by_layer": sink,
    }


# -- continuous batching on the real model ---------------------------------------------------------


@torch.inference_mode()
def continuous_batching(model: CausalLM, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    cache = PagedKVCache(
        model.config,
        num_blocks=cfg["num_blocks"],
        block_size=cfg["block_size"],
        dtype=model.lm_head.weight.dtype,
        device="cuda",
    )
    batcher = ContinuousBatcher(model, cache, max_batch=cfg["max_batch"])
    for i in range(cfg["num_requests"]):
        prompt = [rng.randrange(model.config.vocab_size) for _ in range(rng.randint(*cfg["prompt_len"]))]
        params = SamplingParams(max_new_tokens=rng.randint(*cfg["max_new_tokens"]))
        batcher.submit(Request(id=i, prompt=prompt, params=params, arrival_step=i * cfg["arrival_every"]))
    finished = batcher.run()
    return {
        "num_blocks": cfg["num_blocks"],
        "block_size": cfg["block_size"],
        "max_batch": cfg["max_batch"],
        "steps": batcher.step_count,
        "tokens_generated": sum(len(out) for out in finished.values()),
        "log": batcher.log,
    }


# -- the whole milestone ---------------------------------------------------------------------------


def run_m1(
    config: dict[str, Any], prompts: dict[str, Any], model_dir: str, *, git: dict | None, config_path: str
) -> list[dict[str, Any]]:
    run_id, env, seed = new_run_id(), environment_info(), config["seed"]
    records: list[dict[str, Any]] = []

    def add(experiment: str, metrics: dict[str, Any]) -> None:
        records.append(
            make_record(
                experiment, metrics, run_id=run_id, config={"path": config_path, **config}, git=git, env=env
            )
        )

    dtype = getattr(torch, config["dtype"])
    model = load_pretrained(model_dir, device="cuda", dtype=dtype)
    torch.manual_seed(seed)

    for implementation in ("eager", "sdpa"):
        for row in hf_parity(model_dir, model, prompts["parity"], attn_implementation=implementation):
            add("hf_parity", row)

    d = config["decode"]
    for batch in d["batch_sizes"]:
        add(
            "decode_speed",
            decode_speed(model, batch=batch, prompt_len=d["prompt_len"], steps=d["steps"], seed=seed),
        )
        torch.cuda.empty_cache()

    for length in config["prefill"]["lengths"]:
        add(
            "prefill_speed",
            prefill_speed(model, length=length, repeats=config["prefill"]["repeats"], seed=seed),
        )

    # One decode step and one prefill, profiled and split into components.
    p = config["profile"]
    b, n = p["decode_batch"], d["prompt_len"]
    cache = ContiguousKVCache(model.config, max_batch=b, max_len=n + 64, dtype=dtype, device="cuda")
    with torch.inference_mode():
        model(
            _random_ids(model.config.vocab_size, (b, n), seed),
            torch.arange(n, device="cuda").expand(b, -1),
            cache,
        )
    token = _random_ids(model.config.vocab_size, (b, 1), seed + 1)
    position, zeros = torch.full((b, 1), n, device="cuda"), torch.zeros(b, dtype=torch.long, device="cuda")

    def decode_once():
        with torch.inference_mode():
            return model(token, position, cache, select=zeros).argmax(-1)

    def make_prefill(length: int):
        ids = _random_ids(model.config.vocab_size, (1, length), seed)
        prefill_cache = ContiguousKVCache(
            model.config, max_batch=1, max_len=length, dtype=dtype, device="cuda"
        )
        positions, last = torch.arange(length, device="cuda")[None], torch.tensor([length - 1], device="cuda")

        def prefill_once():
            with torch.inference_mode():
                return model(ids, positions, prefill_cache, select=last)

        return prefill_once

    add("kernel_profile", kernel_profile(decode_once, label=f"decode, batch {b}"))
    add("component_times", component_times(model, decode_once, label=f"decode, batch {b}"))
    for length in p["prefill_lens"]:
        prefill_once = make_prefill(length)
        add("kernel_profile", kernel_profile(prefill_once, label=f"prefill, {length} tokens"))
        add("component_times", component_times(model, prefill_once, label=f"prefill, {length} tokens"))
        torch.cuda.empty_cache()

    from transformers import AutoTokenizer

    ids = AutoTokenizer.from_pretrained(model_dir)(prompts["attention"], return_tensors="pt").input_ids.cuda()
    a = config["attention_maps"]
    add("attention_maps", attention_maps(model, ids, a["layers"], a["head"]))
    add("continuous_batching", continuous_batching(model, config["continuous_batching"], seed))
    return records
