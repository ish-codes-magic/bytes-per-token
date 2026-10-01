"""Parsing vLLM's Prometheus metrics, and sampling them from a fake /metrics endpoint."""

import asyncio

import pytest

from fastserve.serving.server_metrics import SERIES, parse_prometheus

# Shaped like vLLM's export: HELP/TYPE comments, labels, a histogram's _bucket/_count/_sum samples.
EXPORT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-0.6B"} 12.0
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-0.6B"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen3-0.6B"} 0.25
vllm:generation_tokens_total{engine="0",model_name="Qwen/Qwen3-0.6B"} 1500.0
vllm:generation_tokens_created{engine="0",model_name="Qwen/Qwen3-0.6B"} 1.7e+09
vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="Qwen/Qwen3-0.6B"} 4.0
vllm:iteration_tokens_total_count{engine="0",model_name="Qwen/Qwen3-0.6B"} 90.0
vllm:iteration_tokens_total_sum{engine="0",model_name="Qwen/Qwen3-0.6B"} 2100.0
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen3-0.6B"} 4096.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen3-0.6B"} 3072.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="Qwen/Qwen3-0.6B"} 7.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="Qwen/Qwen3-0.6B"} 2.0
process_open_fds 42 1700000000000
"""


def test_parse_reads_values_and_sums_label_sets():
    values = parse_prometheus(EXPORT)
    assert values["vllm:num_requests_running"] == 12
    assert values["vllm:kv_cache_usage_perc"] == 0.25
    assert values["vllm:iteration_tokens_total_count"] == 90
    assert values["vllm:request_success_total"] == 9  # two finish reasons, summed
    assert values["process_open_fds"] == 42  # a trailing timestamp is not the value
    assert not any(name.startswith("#") for name in values)


def test_every_series_is_in_the_sample_export_except_what_it_omits():
    values = parse_prometheus(EXPORT)
    missing = {column for column, name in SERIES.items() if name not in values}
    assert missing == {"prompt_tokens", "preemptions"}  # absent here: the sampler records None for them


def test_sampler_builds_a_timeline_until_stopped():
    web = pytest.importorskip("aiohttp.web")
    import aiohttp

    from fastserve.serving.server_metrics import sample_server

    polls = 0

    async def metrics(request):
        nonlocal polls
        polls += 1
        return web.Response(text=EXPORT.replace("1500.0", f"{1500 + 10 * polls}.0"))

    async def main():
        app = web.Application()
        app.router.add_get("/metrics", metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        stop = asyncio.Event()
        try:
            async with aiohttp.ClientSession() as session:
                task = asyncio.create_task(sample_server(session, f"http://127.0.0.1:{port}", stop, 0.01))
                await asyncio.sleep(0.1)
                stop.set()
                return await task
        finally:
            await runner.cleanup()

    timeline = asyncio.run(main())
    cols = {name: i for i, name in enumerate(timeline["columns"])}
    rows = timeline["rows"]
    assert len(rows) >= 3 and rows[-1][cols["t"]] > rows[0][cols["t"]]
    generated = [row[cols["generation_tokens"]] for row in rows]
    assert generated == sorted(generated) and generated[-1] - generated[0] == 10 * (len(rows) - 1)
    assert all(row[cols["preemptions"]] is None for row in rows)
