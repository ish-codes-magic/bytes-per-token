"""What the engine sees: vLLM's Prometheus metrics, sampled while a load runs. Parsing is stdlib only.

The client measures requests; only the server knows the batch. Its `/metrics` endpoint exposes how many
sequences run and wait, how full the KV cache is, and running totals of tokens and engine steps. Sampling it a
few times per second turns those totals into a timeline:
    output tokens/s = Δ generation tokens ÷ Δt
    step time       = Δt ÷ Δ engine steps
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

# Timeline column -> the Prometheus sample it reads (vLLM 0.30 names; counters gain `_total` when exported).
SERIES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",  # fraction of KV-cache blocks in use, 0 to 1
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "steps": "vllm:iteration_tokens_total_count",  # a histogram of tokens per engine step: its count = steps
    "preemptions": "vllm:num_preemptions_total",
    "prefix_queries": "vllm:prefix_cache_queries_total",  # prompt tokens looked up in the prefix cache
    "prefix_hits": "vllm:prefix_cache_hits_total",  # ... and how many of them were found there
    "spec_drafts": "vllm:spec_decode_num_drafts_total",  # speculative decoding: verification rounds
    "spec_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",  # ... tokens proposed
    "spec_accepted": "vllm:spec_decode_num_accepted_tokens_total",  # ... and accepted
}
SPEC_PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"  # labeled by draft position


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus text format -> {sample name: value}, summing samples that differ only in their labels."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "{" in line:  # name{label="value",...} value [timestamp]
            name, rest = line.split("{", 1)
            fields = rest[rest.rindex("}") + 1 :].split()
        else:  # name value [timestamp]
            name, *fields = line.split()
        try:
            values[name] = values.get(name, 0.0) + float(fields[0])
        except (IndexError, ValueError):
            continue
    return values


def labeled(text: str, name: str, label: str) -> dict[str, float]:
    """One metric's samples keyed by one label's value, e.g. accepted draft tokens by `position`."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        labels, _, rest = line[len(name) + 1 :].rpartition("}")
        for pair in labels.split(","):
            key, _, value = pair.partition("=")
            if key.strip() == label:
                with contextlib.suppress(IndexError, ValueError):
                    values[value.strip('"')] = values.get(value.strip('"'), 0.0) + float(rest.split()[0])
    return values


async def accepted_per_position(session, base_url: str) -> list[float]:
    """vLLM's running count of accepted draft tokens at each draft position (empty without speculation)."""
    async with session.get(f"{base_url}/metrics") as response:
        by_position = labeled(await response.text(), SPEC_PER_POSITION, "position")
    return [by_position[k] for k in sorted(by_position, key=int)]


async def sample_server(
    session, base_url: str, stop: asyncio.Event, interval_s: float = 0.25
) -> dict[str, Any]:
    """Poll /metrics every `interval_s` until `stop` is set, then once more.

    Returns a columnar table: `t` (seconds since the first poll, at the middle of each request) plus one
    column per SERIES entry (None if this vLLM version doesn't export it).
    """
    import aiohttp

    rows: list[list[float | None]] = []
    t0 = time.perf_counter()
    while True:
        before = time.perf_counter()
        try:
            async with session.get(f"{base_url}/metrics") as response:
                values = parse_prometheus(await response.text())
            t = (before + time.perf_counter()) / 2 - t0
            rows.append([t, *(values.get(name) for name in SERIES.values())])
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass  # a missed sample is a gap in the timeline, not a failed run
        if stop.is_set():
            break
        with contextlib.suppress(asyncio.TimeoutError):  # woken early when the load finishes
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
    return {"columns": ["t", *SERIES], "rows": rows}
