"""M2: serving baselines. Start each server configuration, drive every workload at every load point.

One record per (server, workload, load point): the summary metrics plus a compact per-request timing table.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastserve.engine.loader import model_dir
from fastserve.results import environment_info, make_record, new_run_id
from fastserve.serving.client import run_load
from fastserve.serving.metrics import request_rows, summarize
from fastserve.serving.server import VLLMServer
from fastserve.serving.workloads import RequestSpec, Workload

_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def _kv_cache_tokens(server: VLLMServer) -> int | None:
    """vLLM logs how many tokens of KV cache fit after loading the weights: record it with every run."""
    match = _KV_TOKENS.search(server.log_path.read_text(errors="replace"))
    return int(match.group(1).replace(",", "")) if match else None


def _warm_up(server: VLLMServer, model: str) -> None:
    """A few requests before measuring: the first ones pay one-time costs (lazy init, cold caches)."""
    specs = [RequestSpec(id=i, prompt=[1000 + i] * 64, max_tokens=16) for i in range(8)]
    asyncio.run(run_load(server.url, model, specs, {"mode": "closed", "concurrency": 4}))


def run_serving(
    config: dict[str, Any], workloads: dict[str, Any], *, git: dict | None, config_path: str
) -> list[dict[str, Any]]:
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
        model, label, args = server_cfg["model"], server_cfg["label"], server_cfg.get("args", [])
        with VLLMServer(model_dir(model), served_name=model, extra_args=args) as server:
            add(
                "server_start",
                {
                    "label": label,
                    "model": model,
                    "args": args,
                    "startup_s": server.startup_s,
                    "kv_cache_tokens": _kv_cache_tokens(server),
                },
            )
            _warm_up(server, model)
            for name, loads in config["loads"].items():
                specs = Workload.from_config(name, workloads[name]).requests()
                for load in loads:
                    subset = specs[: load.get("requests", len(specs))]
                    results = asyncio.run(run_load(server.url, model, subset, load, seed))
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
