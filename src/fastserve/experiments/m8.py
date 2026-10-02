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
