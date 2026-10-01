"""An async load generator for OpenAI-compatible servers, timestamping every streamed token.

Two ways to apply load:
- open loop: requests arrive on a Poisson schedule whether or not the server keeps up. Overload shows up as
  growing queues and TTFT, which is realistic for public APIs.
- closed loop: N users, each sending the next request when the previous one finishes. Overload slows the
  arrivals down, which hides it; that's why we report both.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastserve.serving.metrics import RequestResult
from fastserve.serving.workloads import RequestSpec, poisson_arrivals


async def send_request(session, base_url: str, model: str, spec: RequestSpec, t0: float, scheduled: float):
    """One streaming completion. Token counts come from vLLM's per-chunk usage stats."""
    import aiohttp

    result = RequestResult(
        id=spec.id, prompt_len=len(spec.prompt), max_tokens=spec.max_tokens, scheduled=scheduled
    )
    payload = {
        "model": model,
        "prompt": spec.prompt,  # token ids: no tokenizer in the loop
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "ignore_eos": spec.ignore_eos,  # vLLM extension: random-token workloads force exactly max_tokens
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    result.sent = time.perf_counter() - t0
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {(await response.text())[:200]}"
                return result
            seen = 0
            async for raw in response.content:  # server-sent events, one "data: {...}" line per chunk
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                now = time.perf_counter() - t0
                usage = json.loads(data).get("usage") or {}
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                if cached is not None:  # only with --enable-prompt-tokens-details (M5's prefix caching)
                    result.cached_tokens = cached
                tokens = usage.get("completion_tokens")
                if tokens is None or tokens <= seen:
                    continue  # e.g. the final usage-only chunk
                if result.first_token is None:
                    result.first_token = now
                result.chunk_times.append(now)
                result.chunk_tokens.append(tokens - seen)
                seen = tokens
            result.output_tokens = seen
            result.finished = result.chunk_times[-1] if result.chunk_times else None
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as err:
        result.error = f"{type(err).__name__}: {err}"
    return result


async def run_load(
    base_url: str, model: str, specs: list[RequestSpec], load: dict[str, Any], seed: int = 0
) -> list[RequestResult]:
    """Send every request. load = {"mode": "open", "rate": r} or {"mode": "closed", "concurrency": n}."""
    import aiohttp

    connector = aiohttp.TCPConnector(limit=0)  # no client-side cap on concurrent connections
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t0 = time.perf_counter()
        if load["mode"] == "open":
            arrivals = poisson_arrivals(load["rate"], len(specs), seed)

            async def fire(spec: RequestSpec, at: float) -> RequestResult:
                await asyncio.sleep(max(0.0, at - (time.perf_counter() - t0)))
                return await send_request(session, base_url, model, spec, t0, scheduled=at)

            return list(await asyncio.gather(*(fire(s, a) for s, a in zip(specs, arrivals, strict=True))))

        if load["mode"] == "closed":
            queue: asyncio.Queue[RequestSpec] = asyncio.Queue()
            for spec in specs:
                queue.put_nowait(spec)
            results: list[RequestResult] = []

            async def user() -> None:
                while not queue.empty():
                    spec = queue.get_nowait()
                    results.append(
                        await send_request(session, base_url, model, spec, t0, time.perf_counter() - t0)
                    )

            await asyncio.gather(*(user() for _ in range(load["concurrency"])))
            return sorted(results, key=lambda r: r.id)
        raise ValueError(f"unknown load mode {load['mode']!r}")
