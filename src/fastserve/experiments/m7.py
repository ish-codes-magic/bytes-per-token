"""M7: the profile that justifies each kernel, their microbenchmarks, and both kernels inside nanoserve.

Tasks (benchmarks/configs/m7_kernels.yaml, `tasks`, one container each):
- profile:    nanoserve's decode step at growing context: how much of it is attention over the KV cache
- ops:        vLLM's norm and INT8-quantization ops, timed apart: the two launches kernel 1 fuses
- norm_quant: kernel 1 against the PyTorch reference and vLLM's ops, eagerly and inside a CUDA graph
- tune:       kernel 2's runtime over (tokens per program × num_warps): the autotuning landscape
- attention:  kernel 2 against nanoserve's fused attention, FlashInfer, and the unfused reference
- nanoserve:  the real model's decode step through each cache, and one step's kernel timeline
- quality:    KL and needle recall with real codes read by the real kernel

Microbenchmarks that involve vLLM's ops or FlashInfer run in the serving image; everything with the model
runs in the research image, like M1–M6's nanoserve numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from fastserve.engine.attention import attention_sdpa
from fastserve.engine.config import ModelConfig
from fastserve.engine.kv_cache import ContiguousKVCache, KVCache
from fastserve.engine.model import CausalLM, RMSNorm, use_fused_attention
from fastserve.integration.kv_cache import QuantizedKVCache, reference_decode
from fastserve.integration.norm_quant import quantize_norm_outputs
from fastserve.kernels import reference
from fastserve.kernels.reference import QuantKV
from fastserve.quant.w8a8 import W8A8Config, activation_quantizer
from fastserve.timing import cuda_graph_time_ms, cuda_time_ms, summarize

Timed = Callable[[], object]


def _stats(times_ms: list[float]) -> dict[str, Any]:
    stats = summarize(times_ms)
    return {"ms": stats.median_ms, "timing": stats.to_dict()}


def _measure(fn: Timed, timing: dict[str, Any], *, flush: int = 0) -> dict[str, Any]:
    """Median and spread of `fn`, or the error that stopped it: a contender that fails is recorded too."""
    try:
        return _stats(cuda_time_ms(fn, warmup=timing["warmup"], iters=timing["iters"], flush_l2_bytes=flush))
    except Exception as err:  # e.g. out of memory at the largest shape, or an unsupported dtype
        torch.cuda.empty_cache()
        return {"error": f"{type(err).__name__}: {str(err)[:200]}"}


def _measure_graph(fn: Timed, timing: dict[str, Any]) -> dict[str, Any]:
    try:
        return _stats(cuda_graph_time_ms(fn, calls=timing["graph_calls"], iters=timing["iters"]))
    except Exception as err:
        torch.cuda.empty_cache()
        return {"error": f"{type(err).__name__}: {str(err)[:200]}"}


def _rand(*shape: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda").to(dtype)


# ---- kernel 1: RMSNorm + INT8 quantization ---------------------------------------------------------------


def norm_quant_contenders(
    x: torch.Tensor, weight: torch.Tensor, eps: float, names: list[str]
) -> dict[str, tuple[Timed, int | None]]:
    """name → (function, bytes it moves). x [rows, d] and weight [d] share a 16-bit dtype."""
    rows, d = x.shape
    e = x.element_size()
    quantized = rows * (d * e + d + 4)  # read the floats, write codes and one float32 scale per row
    found: dict[str, tuple[Timed, int | None]] = {}

    if "torch" in names:  # nanoserve's own modules: a dozen small kernels, each a pass over the data
        norm = RMSNorm(d, eps).to(x.device, x.dtype)
        norm.weight.data = weight
        quantize = activation_quantizer(W8A8Config("int8"))
        found["torch"] = (lambda: quantize(norm(x)), None)

    if any(name.startswith("vllm") for name in names):
        import vllm._custom_ops as ops

        normed = torch.empty_like(x)
        ops.rms_norm(normed, x, weight, eps)  # so "vllm-quant" alone has real input
        vllm: dict[str, tuple[Timed, int]] = {
            "vllm-norm": (lambda: ops.rms_norm(normed, x, weight, eps), rows * 2 * d * e),
            "vllm-quant": (lambda: ops.scaled_int8_quant(normed), quantized),
            "vllm-separate": (
                lambda: (ops.rms_norm(normed, x, weight, eps), ops.scaled_int8_quant(normed)),
                rows * 2 * d * e + quantized,
            ),
            "vllm-fused-fp8": (  # vLLM's own fused op: FP8 codes instead of INT8, the same bytes
                lambda: ops.rms_norm_dynamic_per_token_quant(x, weight, eps, torch.float8_e4m3fn),
                quantized,
            ),
        }
        found.update({name: vllm[name] for name in names if name in vllm})

    if "triton-fused" in names:
        from fastserve.kernels.norm_quant import rms_norm_int8

        out = (
            torch.empty((rows, d), dtype=torch.int8, device=x.device),
            torch.empty(rows, dtype=torch.float32, device=x.device),
        )
        found["triton-fused"] = (lambda: rms_norm_int8(x, weight, eps, out=out), quantized)
    return found


def _code_agreement(codes: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    diff = (codes.int() - want.int()).abs()
    return {"max_code_diff": int(diff.max()), "codes_differing": (diff > 0).float().mean().item()}


@torch.inference_mode()
def norm_quant_bench(spec: dict[str, Any], timing: dict[str, Any]) -> list[dict[str, Any]]:
    """Every contender at every (rows, width): eagerly, and replayed from a CUDA graph."""
    dtype, eps, results = getattr(torch, spec["dtype"]), 1e-6, []
    for d in spec["widths"]:
        weight = (_rand(d, seed=1).abs() + 0.5).to(dtype)
        for rows in spec["rows"]:
            x = _rand(rows, d, seed=rows, dtype=dtype) * 3
            row: dict[str, Any] = {"rows": rows, "d": d, "dtype": spec["dtype"], "contenders": {}}
            for name, (fn, moved) in norm_quant_contenders(x, weight, eps, spec["contenders"]).items():
                row["contenders"][name] = {
                    "bytes_moved": moved,
                    "eager": _measure(fn, timing),
                    "graph": _measure_graph(fn, timing),
                }
            if "triton-fused" in spec["contenders"]:
                from fastserve.kernels.norm_quant import rms_norm_int8

                codes, _ = rms_norm_int8(x, weight, eps)
                row["vs_reference"] = _code_agreement(codes, reference.rms_norm_int8(x, weight, eps)[0])
                if "vllm-separate" in row["contenders"]:  # vLLM rounds the norm to BF16 before quantizing
                    import vllm._custom_ops as ops

                    normed = torch.empty_like(x)
                    ops.rms_norm(normed, x, weight, eps)
                    row["vs_vllm"] = _code_agreement(codes, ops.scaled_int8_quant(normed)[0])
            results.append(row)
            print(f"norm+quant rows {rows} d {d}: " + _brief(row["contenders"], "eager"), flush=True)
    return results


@torch.inference_mode()
def norm_quant_warps(spec: dict[str, Any], timing: dict[str, Any]) -> list[dict[str, Any]]:
    """Kernel 1's only tuning knob: warps per program, at each width."""
    from fastserve.kernels.norm_quant import rms_norm_int8

    dtype, results = getattr(torch, spec["dtype"]), []
    for d in spec["widths"]:
        weight = (_rand(d, seed=1).abs() + 0.5).to(dtype)
        for rows in spec["tune_rows"]:
            x = _rand(rows, d, seed=rows, dtype=dtype) * 3
            out = (
                torch.empty((rows, d), dtype=torch.int8, device="cuda"),
                torch.empty(rows, dtype=torch.float32, device="cuda"),
            )
            for warps in spec["num_warps"]:

                def fn(warps=warps, x=x, weight=weight, out=out):
                    return rms_norm_int8(x, weight, 1e-6, out=out, num_warps=warps)

                results.append({"rows": rows, "d": d, "num_warps": warps, **_measure(fn, timing)})
    return results


def _brief(contenders: dict[str, Any], key: str | None = None) -> str:
    def ms(entry: dict[str, Any]) -> str:
        entry = entry[key] if key else entry
        return f"{entry['ms'] * 1e3:.0f} µs" if "ms" in entry else "failed"

    return ", ".join(f"{name} {ms(entry)}" for name, entry in contenders.items())


# ---- kernel 2: decode attention over a quantized cache ---------------------------------------------------


def random_kv(batch: int, kv_heads: int, tokens: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 keys and values [B, Hkv, T, D]; the keys have an outlier channel, as Qwen3's do (M5)."""
    k = torch.empty((batch, kv_heads, tokens, d), dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    for b in range(batch):  # row by row: the float32 temporaries stay small
        keys = _rand(kv_heads, tokens, d, seed=seed + 2 * b)
        keys[..., 3] += 20.0
        k[b], v[b] = keys, _rand(kv_heads, tokens, d, seed=seed + 2 * b + 1)
    return k, v


def quantize_rows(k: torch.Tensor, v: torch.Tensor, bits: int) -> QuantKV:
    """`reference.quantize_kv`, one sequence at a time, so a large cache fits next to its temporaries."""
    if bits == 16:
        return reference.quantize_kv(k, v, 16)
    rows = [reference.quantize_kv(k[b : b + 1], v[b : b + 1], bits) for b in range(k.shape[0])]
    fields = ("k_codes", "v_codes", "k_scale", "k_zero", "v_scale", "v_zero")
    return QuantKV(bits, rows[0].group, **{f: torch.cat([getattr(r, f) for r in rows]) for f in fields})


def cache_bytes(kv: QuantKV, batch: int, tokens: int) -> int:
    return int(kv.bytes_per_token() * batch * tokens)


@torch.inference_mode()
def attention_case(
    cfg: ModelConfig, batch: int, context: int, spec: dict[str, Any], timing: dict[str, Any]
) -> dict[str, Any]:
    """One (batch, context) shape of one layer's decode attention, through every contender."""
    from fastserve.kernels.kv_attention import attend, default_split

    hq, hkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    scale, flush = d**-0.5, timing["flush_l2_bytes"]
    k, v = random_kv(batch, hkv, context, d, seed=batch + context)
    q = _rand(batch, hq, d, seed=7, dtype=torch.bfloat16)
    lengths = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    caches = {bits: quantize_rows(k, v, bits) for bits in spec["bits"]}
    bf16_bytes = batch * context * hkv * d * 2 * 2
    checked = batch * context <= spec["reference_max_tokens"]  # the float32 reference needs 4× the cache
    exact = reference.partial_attention(q, k, v, lengths, scale)[0] if checked else None  # no quantization

    row: dict[str, Any] = {"batch": batch, "context": context, "contenders": {}}

    def add(
        name: str, fn: Callable[[], torch.Tensor], nbytes: int, want: torch.Tensor | None = None, **extra
    ):
        entry = {"cache_bytes": nbytes, **extra, **_measure(fn, timing, flush=flush)}
        if "ms" in entry:
            entry["gbps"] = nbytes / (entry["ms"] / 1e3) / 1e9
            # The same call without the flush. If its data fits in L2 and it gets no faster, memory was not
            # what limited it.
            entry["warm_ms"] = _measure(fn, timing).get("ms")
            got = fn().float().reshape(batch, hq, d)
            if exact is not None:  # how far from attention over the unquantized cache
                entry["error_vs_bf16"] = ((got - exact).norm() / exact.norm()).item()
            if want is not None:  # how far from the reference reading the same codes
                entry["error_vs_reference"] = ((got - want).abs().max() / want.abs().max()).item()
        row["contenders"][name] = entry

    if "sdpa" in spec["contenders"]:  # nanoserve's decode attention before M7
        mask = torch.ones((batch, 1, 1, context), dtype=torch.bool, device="cuda")
        add("sdpa-bf16", lambda: attention_sdpa(q[:, :, None], k, v, mask, scale), bf16_bytes)

    if "flashinfer" in spec["contenders"] and batch == 1:  # vLLM's attention library, one request
        import flashinfer

        k_nhd, v_nhd = (x[0].transpose(0, 1).contiguous().half() for x in (k, v))  # [T, Hkv, D]
        q_fi = q[0].half()
        add(
            "flashinfer-fp16",
            lambda: flashinfer.single_decode_with_kv_cache(q_fi, k_nhd, v_nhd, sm_scale=scale),
            bf16_bytes,
        )

    for bits, kv in caches.items():
        want = None
        if checked:
            want = reference.merge_partials(*reference.decode_attention(q, kv, lengths, scale))
        if bits != 16 and "dequant" in spec["contenders"] and checked:  # the unfused read of the same codes
            add(
                f"dequant-int{bits}",
                lambda kv=kv: reference.merge_partials(*reference.decode_attention(q, kv, lengths, scale)),
                cache_bytes(kv, batch, context),
            )
        name = "triton-bf16" if bits == 16 else f"triton-int{bits}"
        add(
            name,
            lambda kv=kv: attend(q, kv, lengths, scale, tokens=context),
            cache_bytes(kv, batch, context),
            want,
            split=default_split(batch, hkv, context),
        )
    print(f"attention batch {batch} context {context}: " + _brief(row["contenders"]), flush=True)
    return row


@torch.inference_mode()
def attention_tune(cfg: ModelConfig, spec: dict[str, Any], timing: dict[str, Any]) -> list[dict[str, Any]]:
    """Kernel 2's runtime over (tokens per program × warps per program), per shape and format."""
    from fastserve.kernels.kv_attention import attend

    hq, hkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    quick = {**timing, "iters": spec["iters"]}
    results = []
    for batch, context in spec["shapes"]:
        k, v = random_kv(batch, hkv, context, d, seed=batch + context)
        q = _rand(batch, hq, d, seed=7, dtype=torch.bfloat16)
        lengths = torch.full((batch,), context, dtype=torch.int32, device="cuda")
        for bits in spec["bits"]:
            kv = quantize_rows(k, v, bits)
            for split in [s for s in spec["splits"] if s < context] + [context]:  # last: no splitting
                for warps in spec["num_warps"]:

                    def fn(kv=kv, split=split, warps=warps, q=q, lengths=lengths, context=context):
                        return attend(q, kv, lengths, d**-0.5, tokens=context, split=split, num_warps=warps)

                    measured = _measure(fn, quick, flush=timing["flush_l2_bytes"])
                    results.append(
                        {
                            "batch": batch,
                            "context": context,
                            "bits": bits,
                            "split": split,
                            "num_warps": warps,
                            "programs": batch * hkv * -(-context // split),
                            "cache_bytes": cache_bytes(kv, batch, context),
                            **measured,
                        }
                    )
            best = min(
                (r for r in results if (r["batch"], r["context"], r["bits"]) == (batch, context, bits)),
                key=lambda r: r.get("ms", float("inf")),
            )
            print(
                f"tune batch {batch} context {context} int{bits}: best split {best['split']} "
                f"warps {best['num_warps']} at {best.get('ms', float('nan')) * 1e3:.0f} µs",
                flush=True,
            )
        del k, v
        torch.cuda.empty_cache()
    return results


# ---- the real model ----------------------------------------------------------------------------------------


def make_cache(kind: str, model: CausalLM, batch: int, max_len: int) -> KVCache:
    """bf16-sdpa (nanoserve before M7), or a QuantizedKVCache: {bf16,int8,int4}-kernel, int4-dequant."""
    weight = model.lm_head.weight
    if kind == "bf16-sdpa":
        return ContiguousKVCache(
            model.config, max_batch=batch, max_len=max_len, dtype=weight.dtype, device=weight.device
        )
    fmt, reader = kind.split("-")
    decode = reference_decode
    if reader == "kernel":
        from fastserve.kernels.kv_attention import decode_attention as decode
    return QuantizedKVCache(
        model.config,
        max_batch=batch,
        max_len=max_len,
        bits={"bf16": 16, "int8": 8, "int4": 4}[fmt],
        dtype=weight.dtype,
        device=weight.device,
        decode=decode,
    )


def prefill(model: CausalLM, cache: KVCache, ids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Run ids [B, T] through the cache in chunks. Returns the logits after the last token: [B, vocab]."""
    b, n = ids.shape
    for start in range(0, n, chunk):
        piece = ids[:, start : start + chunk]
        positions = torch.arange(start, start + piece.shape[1], device=ids.device).expand(b, -1)
        last = torch.full((b,), piece.shape[1] - 1, device=ids.device)
        logits = model(piece, positions, cache, select=last)
    return logits


def decode_stepper(model: CausalLM, cache: KVCache, tokens: torch.Tensor, position: int) -> Timed:
    """A function that runs one greedy decode step per call, feeding each step's tokens to the next."""
    b = tokens.shape[0]
    state = {"tokens": tokens, "position": torch.full((b, 1), position, device=tokens.device)}
    zeros = torch.zeros(b, dtype=torch.long, device=tokens.device)

    def step() -> None:
        state["tokens"] = model(state["tokens"], state["position"], cache, select=zeros).argmax(
            -1, keepdim=True
        )
        state["position"] = state["position"] + 1

    return step


@contextmanager
def _timed_attention(model: CausalLM):
    """Record a CUDA event pair around every layer's attention call (the kernel only, not the projections)."""
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def timed(fn):
        def run(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            pairs.append((start, end))
            return out

        return run

    originals = [layer.self_attn.attend for layer in model.model.layers]
    for layer in model.model.layers:
        layer.self_attn.attend = timed(layer.self_attn.attend)
    try:
        yield pairs
    finally:
        for layer, original in zip(model.model.layers, originals, strict=True):
            layer.self_attn.attend = original


def _random_ids(model: CausalLM, batch: int, tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.config.vocab_size, (batch, tokens), generator=g).cuda()


@torch.inference_mode()
def profile_case(model: CausalLM, batch: int, context: int, steps: int, seed: int) -> dict[str, Any]:
    """nanoserve's decode step at this context, before M7: its time, and attention's share of it."""
    from fastserve.experiments.m1 import kernel_profile

    cfg = model.config
    cache = make_cache("bf16-sdpa", model, batch, context + 3 * steps + 16)
    tokens = prefill(model, cache, _random_ids(model, batch, context, seed)).argmax(-1, keepdim=True)
    step = decode_stepper(model, cache, tokens, context)
    total = summarize(cuda_time_ms(step, warmup=5, iters=steps))
    with _timed_attention(model) as pairs:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    attention_ms = sum(start.elapsed_time(end) for start, end in pairs) / steps
    kernels = kernel_profile(step, label=f"decode, batch {batch}, context {context}")
    return {
        "batch": batch,
        "context": context,
        "step_ms": total.median_ms,
        "timing": total.to_dict(),
        "attention_ms": attention_ms,
        "attention_share": attention_ms / total.median_ms,
        "kernels": kernels["kernels"],
        "kernel_ms": kernels["kernel_ms"],
        "kv_bytes": cfg.kv_bytes_per_token() * context * batch,
    }


def kernel_timeline(fn: Timed) -> dict[str, Any]:
    """Every GPU kernel one call of `fn` runs: (name, start µs, duration µs), from PyTorch's profiler."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    names = sorted({e.name for e in kernels})
    index = {name: i for i, name in enumerate(names)}
    origin = kernels[0].time_range.start if kernels else 0
    events = [[index[e.name], e.time_range.start - origin, e.time_range.elapsed_us()] for e in kernels]
    return {"names": [name[:80] for name in names], "events": events}


@torch.inference_mode()
def nanoserve_case(
    model: CausalLM, kind: str, batch: int, context: int, steps: int, seed: int, baseline: torch.Tensor | None
) -> tuple[dict[str, Any], torch.Tensor, Timed]:
    """Decode steps of the real model through one cache. Also returns the first step's logits and the step."""
    cache = make_cache(kind, model, batch, context + steps + 48)
    tokens = prefill(model, cache, _random_ids(model, batch, context, seed)).argmax(-1, keepdim=True)
    zeros = torch.zeros(batch, dtype=torch.long, device="cuda")
    logits = model(tokens, torch.full((batch, 1), context, device="cuda"), cache, select=zeros).float()
    step = decode_stepper(model, cache, logits.argmax(-1, keepdim=True), context + 1)
    stats = summarize(cuda_time_ms(step, warmup=5, iters=steps))
    per_token = (
        cache.bytes_per_token() if isinstance(cache, QuantizedKVCache) else model.config.kv_bytes_per_token()
    )
    row = {
        "cache": kind,
        "batch": batch,
        "context": context,
        "step_ms": stats.median_ms,
        "tokens_per_s": batch / (stats.median_ms / 1e3),
        "timing": stats.to_dict(),
        "kv_bytes_per_token": per_token,
        "kv_bytes": per_token * context * batch,
    }
    if baseline is not None:  # the same prompt through the BF16 cache: how much did the logits move?
        row["top1_agreement"] = (logits.argmax(-1) == baseline.argmax(-1)).float().mean().item()
        row["max_logit_diff"] = (logits - baseline).abs().max().item()
        ref, cand = torch.log_softmax(baseline, -1), torch.log_softmax(logits, -1)
        row["kl_first_step"] = (ref.exp() * (ref - cand)).sum(-1).mean().item()
    return row, logits, step


@torch.inference_mode()
def norm_in_model(model: CausalLM, steps: int, seed: int) -> dict[str, Any]:
    """Kernel 1 inside nanoserve: every pre-norm emits INT8 activations, by the reference or the kernel."""
    originals = [(layer.input_layernorm, layer.post_attention_layernorm) for layer in model.model.layers]
    ids = _random_ids(model, 1, 128, seed)
    row: dict[str, Any] = {"batch": 1, "context": 128}
    logits = {}
    try:
        for name, fused in (("reference", False), ("fused", True)):
            quantize_norm_outputs(model, fused)
            cache = make_cache("bf16-sdpa", model, 1, 128 + steps + 16)
            logits[name] = prefill(model, cache, ids).float()
            step = decode_stepper(model, cache, logits[name].argmax(-1, keepdim=True), 128)
            row[f"{name}_step_ms"] = summarize(cuda_time_ms(step, warmup=5, iters=steps)).median_ms
    finally:
        for layer, (first, second) in zip(model.model.layers, originals, strict=True):
            layer.input_layernorm, layer.post_attention_layernorm = first, second
    row["max_logit_diff"] = (logits["fused"] - logits["reference"]).abs().max().item()
    row["top1_agreement"] = (
        (logits["fused"].argmax(-1) == logits["reference"].argmax(-1)).float().mean().item()
    )
    return row


# ---- quality through the real codes ------------------------------------------------------------------------


def stepwise_logits(model: CausalLM, kind: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """ids [1, T] → logits [1, T, vocab], one token at a time through a fresh cache: every position is a
    decode step, so with a kernel cache every attention goes through the kernel."""

    def run(ids: torch.Tensor) -> torch.Tensor:
        b, n = ids.shape
        cache = make_cache(kind, model, b, n)
        out = [model(ids[:, t : t + 1], torch.full((b, 1), t, device=ids.device), cache) for t in range(n)]
        return torch.cat(out, dim=1)

    return run


def kernel_kl(bench: Any, kinds: list[str]) -> list[dict[str, Any]]:
    """Each cache against the BF16 cache on M2's WikiText-2 windows, decoded token by token."""
    from fastserve.quality.perplexity import evaluate

    model = bench.ref
    memo: dict[bytes, torch.Tensor] = {}
    plain = stepwise_logits(model, "bf16-sdpa")

    def reference_logits(ids: torch.Tensor) -> torch.Tensor:  # the same windows serve every candidate
        key = ids.cpu().numpy().tobytes()
        if key not in memo:
            memo[key] = plain(ids)
        return memo[key]

    rows = []
    for kind in kinds:
        start = time.perf_counter()
        metrics = evaluate(bench.windows, reference_logits, stepwise_logits(model, kind), batch=1)
        rows.append({"model": bench.model_name, "cache": kind, "window": bench.windows.shape[1], **metrics})
        print(f"{kind}: KL {metrics['mean_kl']:.4g} ({time.perf_counter() - start:.0f} s)", flush=True)
    return rows


@torch.inference_mode()
def kernel_needle(bench: Any, kind: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """M2's needle grid with the prompt stored as codes and every answer token decoded by the kernel."""
    from fastserve.engine.generate import generate_long
    from fastserve.engine.sampler import SamplingParams
    from fastserve.quality.needle import grid, passed

    model, tokenizer = bench.ref, bench.tokenizer
    stop = tuple(tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
    params = SamplingParams(max_new_tokens=cfg["max_new_tokens"], stop_token_ids=stop)
    cells = []
    for case in grid(tokenizer, cfg["lengths"], cfg["depths"], cfg["secrets"]):
        prompt = tokenizer.encode(case.prompt, add_special_tokens=False)
        cache = make_cache(kind, model, 1, len(prompt) + params.max_new_tokens)
        output = generate_long(model, prompt, params, chunk=cfg["chunk"], cache=cache)
        answer = tokenizer.decode(output, skip_special_tokens=True)
        cells.append(
            {
                "length": case.context_tokens,
                "depth": case.depth,
                "secret": case.secret,
                "passed": passed(case, answer),
                "answer": answer.strip()[:80],
            }
        )
        del cache
    rate = sum(c["passed"] for c in cells) / len(cells)
    return {"model": bench.model_name, "cache": kind, "cells": cells, "pass_rate": rate}


# ---- one task --------------------------------------------------------------------------------------------


def run_task(
    task: str, config: dict[str, Any], *, run_id: str, git: dict | None, config_path: str, repo: str = "."
) -> list[dict]:
    """Run one section of benchmarks/configs/m7_kernels.yaml's `tasks` in this container."""
    from fastserve.results import environment_info, make_record

    spec, timing, seed = config["tasks"][task], config["timing"], config["seed"]
    meta = {"path": config_path, "task": task, "timing": timing}
    env = environment_info()
    kind = spec["kind"]

    def record(experiment: str, metrics: dict[str, Any]) -> dict:
        return make_record(experiment, metrics, run_id=run_id, config=meta, git=git, env=env)

    def model_config() -> ModelConfig:  # head counts come from config.json, never from this file
        name = config["model"].split("/")[-1]
        return ModelConfig.from_pretrained_json(f"{repo}/benchmarks/models/{name}.config.json")

    if kind == "norm_quant":
        section = {**config["norm_quant"], "contenders": spec["contenders"]}
        records = [record("m7_norm_quant", row) for row in norm_quant_bench(section, timing)]
        if "triton-fused" in spec["contenders"]:
            records += [record("m7_norm_quant_warps", row) for row in norm_quant_warps(section, timing)]
        return records

    if kind == "attention":
        section, cfg = config["attention"], model_config()
        records = []
        for batch in section["batches"]:
            for context in section["contexts"]:
                if batch * context > section["max_tokens"]:
                    skipped = f"{batch} × {context} tokens exceed max_tokens"
                    records.append(
                        record("m7_attention", {"batch": batch, "context": context, "skipped": skipped})
                    )
                    continue
                records.append(record("m7_attention", attention_case(cfg, batch, context, section, timing)))
                torch.cuda.empty_cache()  # the case's tensors died with its frame
        return records

    if kind == "tune":
        rows = attention_tune(model_config(), config["tune"], timing)
        return [record("m7_tune", row) for row in rows]

    # Everything below runs the real model.
    from fastserve.engine.loader import load_pretrained, model_dir

    if kind in ("profile", "nanoserve"):
        model = use_fused_attention(load_pretrained(model_dir(config["model"])))
        section = config[kind]
        if kind == "profile":
            records = []
            for batch, context in section["cases"]:
                row = profile_case(model, batch, context, section["steps"], seed)
                records.append(record("m7_profile", row))
                print(
                    f"profile batch {batch} context {context}: step {row['step_ms']:.1f} ms, attention "
                    f"{row['attention_share']:.0%}, {row['kernels']} kernels",
                    flush=True,
                )
                torch.cuda.empty_cache()
            return records

        records = []
        for batch, context in section["cases"]:
            baseline = None
            for cache_kind in section["caches"]:
                row, logits, step = nanoserve_case(
                    model, cache_kind, batch, context, section["steps"], seed, baseline
                )
                baseline = logits if cache_kind == "bf16-sdpa" else baseline
                records.append(record("m7_nanoserve", {"model": config["model"], **row}))
                if [batch, context] == section["timeline"]["case"] and cache_kind in section["timeline"][
                    "caches"
                ]:
                    timeline = {
                        "cache": cache_kind,
                        "batch": batch,
                        "context": context,
                        **kernel_timeline(step),
                    }
                    records.append(record("m7_timeline", timeline))
                print(
                    f"nanoserve batch {batch} context {context} {cache_kind}: {row['step_ms']:.1f} ms/step",
                    flush=True,
                )
                del step
                torch.cuda.empty_cache()
        records.append(record("m7_norm_in_model", norm_in_model(model, section["steps"], seed)))
        return records

    if kind == "quality":
        from fastserve.experiments.m3 import load_bench

        section = config["quality"]
        bench = load_bench(config["model"], {"eval": section["kl"]})
        use_fused_attention(bench.ref)
        records = [record("m7_kl", row) for row in kernel_kl(bench, section["caches"])]
        for cache_kind in section["needle"]["caches"]:
            start = time.perf_counter()
            row = kernel_needle(bench, cache_kind, section["needle"])
            records.append(record("m7_needle", row))
            print(
                f"needle {cache_kind}: {row['pass_rate']:.2f} ({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
        return records

    raise ValueError(f"unknown task kind {kind!r}")
