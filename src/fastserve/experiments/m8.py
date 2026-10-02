"""M8: what vLLM's FlashInfer backend costs per pass, timed call by call.

On a GPU without FlashInfer's TRTLLM kernels (anything before Hopper), vLLM 0.30 reaches FlashInfer's own
kernels through three paths (vllm/v1/attention/backends/flashinfer.py, `build` and `forward`):

- **graph decode**: one new token per sequence inside a full CUDA graph. `fast_decode_plan` once per pass
  (a few buffer copies), and the attention call is replayed by the graph.
- **eager decode**: one new token per sequence outside a full graph. The decode wrapper's full `plan` once
  per pass, then `run` from Python in every layer.
- **prefill**: several new tokens per sequence. The prefill wrapper's full `plan` once per pass, then `run`
  from Python in every layer. A speculative pass (k + 1 tokens per sequence) takes this path, because the
  backend counts it as a decode only when the TRTLLM kernels exist.

`plan_cost` makes the same calls with the same arguments on a cache of the model's shape and times each one
twice: until the call returns (host time) and until the GPU has finished what it queued (total time).
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from fastserve.engine.config import ModelConfig
from fastserve.results import environment_info, make_record

PAGE = 16  # tokens per KV block, vLLM's default


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "min": ordered[0],
    }


def _time_calls(fn, repeats: int, warmup: int) -> dict[str, dict[str, float]]:
    """Per call, in ms: `host` until the call returns, `total` until the GPU has finished what it queued."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    host, total = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        returned = time.perf_counter()
        torch.cuda.synchronize()
        done = time.perf_counter()
        host.append(1e3 * (returned - start))
        total.append(1e3 * (done - start))
    return {"host_ms": _percentiles(host), "total_ms": _percentiles(total)}


def plan_cost(
    cfg: ModelConfig, batch: int, context: int, new_tokens: int, repeats: int, warmup: int
) -> list[dict[str, Any]]:
    """One row per call kind for `batch` sequences of `context` cached tokens.

    Kinds: "prefill plan" / "prefill run" (`new_tokens` per sequence), "eager decode plan" / "decode run"
    (one token per sequence), "graph decode plan" (`fast_decode_plan`, what a full CUDA graph needs).
    A kind that this FlashInfer version rejects is recorded with its error instead of a time.
    """
    import torch
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
    from flashinfer.decode import fast_decode_plan
    from vllm import envs

    device, dtype = torch.device("cuda"), torch.bfloat16
    heads, kv_heads, head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    scale = head_dim**-0.5
    pages = -(-context // PAGE)
    workspace = torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device=device)

    # The paged cache of one layer, as vLLM hands it over: K and V, [pages, kv_heads, PAGE, head_dim] (HND).
    shape = (batch * pages, kv_heads, PAGE, head_dim)
    cache = (torch.randn(shape, dtype=dtype, device=device), torch.randn(shape, dtype=dtype, device=device))
    # What `build` passes: block indices on the GPU, the per-sequence bookkeeping in pinned host memory.
    indices = torch.arange(batch * pages, dtype=torch.int32, device=device)
    indptr = (torch.arange(batch + 1, dtype=torch.int32) * pages).pin_memory()
    last_page = torch.full((batch,), context - (pages - 1) * PAGE, dtype=torch.int32).pin_memory()
    seq_lens = torch.full((batch,), context, dtype=torch.int32).pin_memory()
    qo_indptr = (torch.arange(batch + 1, dtype=torch.int32) * new_tokens).pin_memory()

    q_prefill = torch.randn(batch * new_tokens, heads, head_dim, dtype=dtype, device=device)
    q_decode = torch.randn(batch, heads, head_dim, dtype=dtype, device=device)
    out_prefill, out_decode = torch.empty_like(q_prefill), torch.empty_like(q_decode)

    prefill = BatchPrefillWithPagedKVCacheWrapper(workspace, "HND", backend="auto")
    decode = BatchDecodeWithPagedKVCacheWrapper(
        workspace, "HND", use_cuda_graph=False, use_tensor_cores=True, backend="auto"
    )
    graph = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "HND",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.zeros(batch + 1, dtype=torch.int32, device=device),
        paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=torch.zeros(batch, dtype=torch.int32, device=device),
        use_tensor_cores=True,
        backend="auto",
    )
    decode_kwargs = {
        "indptr": indptr,
        "indices": indices,
        "last_page_len": last_page,
        "num_qo_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "page_size": PAGE,
        "pos_encoding_mode": "NONE",
        "window_left": -1,
        "logits_soft_cap": None,
        "q_data_type": dtype,
        "kv_data_type": dtype,
        "sm_scale": scale,
        "non_blocking": True,
    }

    def plan_prefill() -> None:
        prefill.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=last_page,
            seq_lens=seq_lens,
            num_qo_heads=heads,
            num_kv_heads=kv_heads,
            head_dim_qk=head_dim,
            page_size=PAGE,
            causal=True,
            sm_scale=scale,
            window_left=-1,
            logits_soft_cap=None,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
        )

    def plan_decode() -> None:
        decode.plan(**decode_kwargs, o_data_type=dtype, seq_lens=seq_lens)

    def plan_graph() -> None:
        fast_decode_plan(graph, **decode_kwargs)

    def first_graph_plan() -> None:  # vLLM's first call on a graph wrapper is the full plan
        graph.plan(**decode_kwargs, o_data_type=dtype, seq_lens=seq_lens)

    calls = [
        ("prefill plan", None, plan_prefill),
        ("prefill run", plan_prefill, lambda: prefill.run(q_prefill, cache, out=out_prefill)),
        ("eager decode plan", None, plan_decode),
        ("decode run", plan_decode, lambda: decode.run(q_decode, cache, out=out_decode)),
        ("graph decode plan", first_graph_plan, plan_graph),
    ]
    rows = []
    for kind, setup, fn in calls:
        row: dict[str, Any] = {"kind": kind, "batch": batch, "context": context, "new_tokens": new_tokens}
        try:
            if setup is not None:
                setup()
            row |= _time_calls(fn, repeats, warmup)
        except Exception as err:  # an argument this FlashInfer version does not take: keep the other kinds
            row["error"] = f"{type(err).__name__}: {str(err)[:300]}"
        rows.append(row)
    return rows


def run_plan_cost(
    config: dict[str, Any], repo: str | Path, *, run_id: str, git: dict | None, config_path: str
) -> list[dict]:
    """Every (batch, context) of the config's `plan_cost` block, one record per call kind."""
    from importlib.metadata import version

    spec = config["plan_cost"]
    name = spec["model"].split("/")[-1]
    cfg = ModelConfig.from_pretrained_json(Path(repo, "benchmarks", "models", f"{name}.config.json"))
    env = environment_info()
    records = []
    for batch in spec["batches"]:
        for context in spec["contexts"]:
            rows = plan_cost(cfg, batch, context, spec["new_tokens"], spec["repeats"], spec["warmup"])
            for row in rows:
                metrics = {"model": spec["model"], "flashinfer": version("flashinfer-python"), **row}
                records.append(
                    make_record(
                        "m8_plan_cost",
                        metrics,
                        run_id=run_id,
                        config={"path": config_path, **config},
                        git=git,
                        env=env,
                    )
                )
    return records


# ---- where a pass's host time goes: vLLM's own profiler ----------------------------------------------------


def self_times(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Self time of every complete ("X") event in a Chrome trace: its duration minus its children's.

    Events on one thread nest (an op inside a Python function inside a step). Sorting by start time, with
    the longer event first on ties, lets a stack of open events find each event's parent. Returns the
    events with `self_us` added; GPU kernels (which run on their own timeline) keep their full duration.
    """
    by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("ph") == "X" and "dur" in event:
            by_thread.setdefault((event.get("pid"), event.get("tid")), []).append(event)
    out = []
    for thread in by_thread.values():
        thread.sort(key=lambda e: (e["ts"], -e["dur"]))
        open_events: list[dict[str, Any]] = []
        for event in thread:
            event = {**event, "self_us": float(event["dur"])}
            while open_events and open_events[-1]["ts"] + open_events[-1]["dur"] <= event["ts"]:
                open_events.pop()
            if open_events:
                open_events[-1]["self_us"] -= event["dur"]
            open_events.append(event)
            out.append(event)
    return out


def summarize_trace(events: list[dict[str, Any]], top: int = 30) -> dict[str, Any]:
    """Per (category, name): calls, self time and total time, for host events and for GPU kernels."""
    timed = self_times(events)
    groups: dict[tuple[str, str], dict[str, float]] = {}
    for event in timed:
        key = (event.get("cat", ""), str(event.get("name", ""))[:140])
        group = groups.setdefault(key, {"calls": 0, "self_ms": 0.0, "total_ms": 0.0})
        group["calls"] += 1
        group["self_ms"] += event["self_us"] / 1e3
        group["total_ms"] += event["dur"] / 1e3
    rows = [{"cat": cat, "name": name, **group} for (cat, name), group in groups.items()]
    gpu = [r for r in rows if r["cat"] in ("kernel", "gpu_memcpy", "gpu_memset")]
    host = [r for r in rows if r["cat"] not in ("kernel", "gpu_memcpy", "gpu_memset")]
    spans = [(e["ts"], e["ts"] + e["dur"]) for e in timed]
    return {
        "span_ms": (max(end for _, end in spans) - min(start for start, _ in spans)) / 1e3 if spans else 0.0,
        "host_self_ms": sum(r["self_ms"] for r in host),
        "gpu_ms": sum(r["total_ms"] for r in gpu),
        "host": sorted(host, key=lambda r: -r["self_ms"])[:top],
        "gpu": sorted(gpu, key=lambda r: -r["total_ms"])[:top],
        "annotations": sorted(
            (r for r in rows if r["cat"] == "user_annotation"), key=lambda r: -r["total_ms"]
        )[:top],
    }
