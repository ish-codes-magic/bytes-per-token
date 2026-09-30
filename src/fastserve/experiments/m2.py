"""M2: serving baselines. Start each server configuration, drive every workload at every load point.

One record per (server, workload, load point): the summary metrics, a compact per-request timing table, and a
timeline of the server's own metrics (running and waiting requests, KV-cache use, tokens, engine steps).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastserve.engine.loader import model_dir
from fastserve.results import environment_info, make_record, new_run_id
from fastserve.serving.client import run_load
from fastserve.serving.metrics import RequestResult, request_rows, summarize
from fastserve.serving.server import VLLMServer
from fastserve.serving.server_metrics import sample_server
from fastserve.serving.workloads import RequestSpec, Workload

# What vLLM decides at startup and prints:
# - how many tokens of KV cache fit, and how much memory the weights and the KV cache take
# - which kernel it picked for each quantized layer type
# (Its per-step token budget isn't in the 0.30 log; the server timeline measures it instead.)
_FROM_LOG = {
    "kv_cache_tokens": (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), int),
    "model_memory_gib": (re.compile(r"Model loading took ([\d.]+) ?GiB"), float),
    "kv_cache_memory_gib": (re.compile(r"Available KV cache memory: ([\d.]+) ?GiB"), float),
}
_KERNEL = re.compile(r"(Using \S*Kernel\S* for \S+|Selected \S*Kernel\S* for \S+)")


def _from_log(server: VLLMServer) -> dict[str, Any]:
    log = server.log_path.read_text(errors="replace")
    found: dict[str, Any] = {}
    for key, (pattern, cast) in _FROM_LOG.items():
        match = pattern.search(log)
        found[key] = cast(match.group(1).replace(",", "")) if match else None
    found["kernels"] = sorted(set(_KERNEL.findall(log)))
    return found


async def _monitored(
    url: str, model: str, specs: list[RequestSpec], load: dict[str, Any], seed: int
) -> tuple[list[RequestResult], dict[str, Any]]:
    """Run one load point while sampling the server's /metrics; the two clocks start within milliseconds."""
    import aiohttp

    stop = asyncio.Event()
    async with aiohttp.ClientSession() as session:
        sampler = asyncio.create_task(sample_server(session, url, stop))
        results = await run_load(url, model, specs, load, seed)
        stop.set()
        return results, await sampler


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

    A server entry may give `path` (a local checkpoint, e.g. a quantized one); otherwise the model comes from
    the Hugging Face cache.
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
        path = server_cfg.get("path") or model_dir(model)
        with VLLMServer(path, served_name=model, extra_args=args) as server:
            add(
                "server_start",
                {
                    "label": label,
                    "model": model,
                    "args": args,
                    "startup_s": server.startup_s,
                    **_from_log(server),
                },
            )
            _warm_up(server, model)
            for name, loads in config["loads"].items():
                specs = Workload.from_config(name, workloads[name]).requests()
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
