"""M2: serving baselines. Start each server configuration, drive every workload at every load point.

One record per (server, workload, load point): the summary metrics, a compact per-request timing table, and a
timeline of the server's own metrics (running and waiting requests, KV-cache use, tokens, engine steps).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from fastserve.engine.loader import checkpoint_dir, model_dir
from fastserve.hw.telemetry import PowerSampler
from fastserve.results import environment_info, make_record, new_run_id
from fastserve.serving.client import run_load
from fastserve.serving.metrics import RequestResult, request_rows, summarize
from fastserve.serving.server import VLLMServer
from fastserve.serving.server_metrics import accepted_per_position, sample_server
from fastserve.serving.workloads import RequestSpec, Workload, text_requests

# What vLLM decides at startup and prints:
# - how many tokens of KV cache fit, and how much memory the weights and the KV cache take
# - which kernel it picked for each quantized layer type
# (Its per-step token budget isn't in the 0.30 log; the server timeline measures it instead.)
_FROM_LOG = {
    "kv_cache_tokens": (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), int),
    "model_memory_gib": (re.compile(r"Model loading took ([\d.]+) ?GiB"), float),
    "kv_cache_memory_gib": (re.compile(r"Available KV cache memory: ([\d.]+) ?GiB"), float),
    # M5: how many full-length requests fit at once, which attention kernel runs, and the cache's dtype
    "max_concurrency": (re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"), float),
    # (an explicit --attention-backend is printed as "AttentionBackendEnum.FLASHINFER")
    "attention_backend": (
        re.compile(r"Using (?:AttentionBackendEnum\.)?(\S+?) (?:attention )?backend", re.I),
        str,
    ),
    "kv_cache_dtype": (re.compile(r"Using (\S+) data type to store kv cache", re.I), str),
}
_KERNEL = re.compile(r"(Using \S*Kernel\S* for \S+|Selected \S*Kernel\S* for \S+)")
# M8: which CUDA graphs were captured (progress bars that reached N/N), and vLLM's warning when the
# attention backend cannot be captured in a full graph for this kind of batch
_GRAPHS = re.compile(r"Capturing ((?:prefill |decode )?)CUDA graphs \((\w+)\):\s+100%[^\r\n]*?\b(\d+)/\3\b")
_FALLBACK = re.compile(r"(CUDAGraphMode\.\w+ is not supported[^\r\n]*?setting cudagraph_mode=\w+)")


def cuda_graphs(log: str) -> dict[str, Any]:
    """The CUDA graphs vLLM captured at startup, counted per kind, and its warning if it had to fall back.

    A full graph replays a whole forward pass as one launch. A piecewise graph covers the layers between
    attention calls, and attention runs from Python in between. With speculative decoding the drafter's
    graphs are printed too ("prefill full", "decode full").
    """
    captured: dict[str, int] = {}
    for who, mode, count in _GRAPHS.findall(log):
        kind = f"{who}{mode}".lower()
        captured[kind] = max(captured.get(kind, 0), int(count))  # the memory profiler captures a few first
    fallback = _FALLBACK.search(log)
    return {"captured": captured, "fallback": fallback.group(1) if fallback else None}


def _from_log(server: VLLMServer) -> dict[str, Any]:
    log = server.log_path.read_text(errors="replace")
    found: dict[str, Any] = {}
    for key, (pattern, cast) in _FROM_LOG.items():
        match = pattern.search(log)
        found[key] = cast(match.group(1).replace(",", "")) if match else None
    found["kernels"] = sorted(set(_KERNEL.findall(log)))
    found["cuda_graphs"] = cuda_graphs(log)
    return found


async def _monitored(
    url: str, model: str, specs: list[RequestSpec], load: dict[str, Any], seed: int
) -> tuple[list[RequestResult], dict[str, Any]]:
    """Run one load point while sampling the server's /metrics; the two clocks start within milliseconds."""
    import aiohttp

    stop = asyncio.Event()
    async with aiohttp.ClientSession() as session:
        before = await accepted_per_position(session, url)
        sampler = asyncio.create_task(sample_server(session, url, stop))
        with PowerSampler() as power:  # the GPU's draw during this load: tokens per joule (M8)
            results = await run_load(url, model, specs, load, seed)
        stop.set()
        timeline = await sampler
        timeline["power"] = power.summary()
        after = await accepted_per_position(session, url)
        if after:  # speculative decoding: draft tokens accepted at each position during this load
            before = before or [0.0] * len(after)
            timeline["spec_accepted_per_position"] = [b - a for a, b in zip(before, after, strict=True)]
        return results, timeline


def _speculative_args(spec: dict[str, Any] | None) -> list[str]:
    """vLLM's --speculative-config for a server's `speculative` block (M6). A drafter given as a Hub id is
    resolved to its local snapshot. The whole repo is fetched, not just the files nanoserve reads: drafter
    heads such as EAGLE-3 ship their weights in other formats (e.g. pytorch_model.bin)."""
    if not spec:
        return []
    spec = dict(spec)
    if "model" in spec and not spec["model"].startswith("/"):
        from huggingface_hub import snapshot_download

        spec["model"] = snapshot_download(spec["model"])
    return ["--speculative-config", json.dumps(spec)]


def _requests(name: str, cfg: dict[str, Any], model_path: str) -> list[RequestSpec]:
    """A workload's requests: seeded random tokens, or (with `tasks`) real prompts through the tokenizer."""
    if "tasks" not in cfg:
        return Workload.from_config(name, cfg).requests()
    from transformers import AutoTokenizer

    from fastserve.quality.prompts import task_prompts

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = {task: task_prompts(tokenizer, task, cfg["prompts_per_task"]) for task in cfg["tasks"]}
    return text_requests(prompts, cfg["output_len"])


def _warm_up(server: VLLMServer, model: str) -> None:
    """A few requests before measuring: the first ones pay one-time costs (lazy init, cold caches)."""
    specs = [RequestSpec(id=i, prompt=[1000 + i] * 64, max_tokens=16) for i in range(8)]
    asyncio.run(run_load(server.url, model, specs, {"mode": "closed", "concurrency": 4}))


def run_serving(
    config: dict[str, Any],
    workloads: dict[str, Any],
    *,
    git: dict | None,
    config_path: str,
    only: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Every server in the config (or just those whose `name` is in `only`), every workload, every load point.

    A server entry may give `path`: a quantized checkpoint, as a Hub id (downloaded once into the cache) or
    a local folder. Otherwise the model itself comes from the Hugging Face cache.
    """
    run_id, env, seed = new_run_id(), environment_info(), config["seed"]
    slo = config["slo"]
    records: list[dict[str, Any]] = []

    def add(experiment: str, metrics: dict[str, Any]) -> None:
        records.append(
            make_record(
                experiment, metrics, run_id=run_id, config={"path": config_path, **config}, git=git, env=env
            )
        )

    for server_cfg in config["servers"]:
        if only is not None and server_cfg.get("name", server_cfg["label"]) not in only:
            continue
        model, label, args = server_cfg["model"], server_cfg["label"], server_cfg.get("args", [])
        path = checkpoint_dir(server_cfg["path"]) if server_cfg.get("path") else model_dir(model)
        env = server_cfg.get("env")
        args = [*args, *_speculative_args(server_cfg.get("speculative"))]
        with VLLMServer(path, served_name=model, extra_args=args, env=env) as server:
            add(
                "server_start",
                {
                    "label": label,
                    "model": model,
                    "args": args,
                    "env": env,
                    "speculative": server_cfg.get("speculative"),
                    "startup_s": server.startup_s,
                    **_from_log(server),
                },
            )
            _warm_up(server, model)
            for name, loads in config["loads"].items():
                if "workloads" in server_cfg and name not in server_cfg["workloads"]:
                    continue  # this server runs only some of the config's workloads (M5)
                specs = _requests(name, workloads[name], path)
                for load in loads:
                    subset = specs[: load.get("requests", len(specs))]
                    results, timeline = asyncio.run(_monitored(server.url, model, subset, load, seed))
                    summary = summarize(
                        results,
                        slo_ttft_s=slo["ttft_ms"] / 1e3,
                        slo_tpot_s=slo["tpot_ms"] / 1e3,
                        dollars_per_hour=config["dollars_per_hour"],
                        seed=seed,
                    )
                    add(
                        "serving",
                        {
                            "server": label,
                            "model": model,
                            "workload": name,
                            "tasks": workloads[name].get(
                                "tasks"
                            ),  # real-prompt workloads: request i is task i mod n
                            "load": load,
                            "summary": summary,
                            "requests": request_rows(results),
                            "server_timeline": timeline,
                        },
                    )
                    print(
                        f"{model} {name} {load}: {summary.get('output_throughput', 0):,.0f} tok/s, "
                        f"TTFT p50 {(summary.get('ttft_ms') or {}).get('p50', float('nan')):.0f} ms, "
                        f"TPOT p50 {(summary.get('tpot_ms') or {}).get('p50', float('nan')):.1f} ms",
                        flush=True,
                    )
    return records


def offline_throughput(
    model_path: str, specs: list[RequestSpec], *, max_num_seqs: int = 256
) -> dict[str, Any]:
    """The engine alone: every request submitted at once to vLLM's offline `LLM`: no HTTP server, no client.

    Compared with the served peak, it separates what the GPU engine can do from the serving overhead.
    """
    import time

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_num_seqs=max_num_seqs, enable_prefix_caching=False, seed=0)
    prompts = [{"prompt_token_ids": s.prompt} for s in specs]
    params = [SamplingParams(max_tokens=s.max_tokens, ignore_eos=True, temperature=0.0) for s in specs]
    llm.generate(prompts[:8], params[:8])  # warm-up
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start
    output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    prompt_tokens = sum(len(s.prompt) for s in specs)
    return {
        "requests": len(specs),
        "max_num_seqs": max_num_seqs,
        "elapsed_s": elapsed,
        "output_tokens": output_tokens,
        "output_throughput": output_tokens / elapsed,
        "total_throughput": (output_tokens + prompt_tokens) / elapsed,
    }
