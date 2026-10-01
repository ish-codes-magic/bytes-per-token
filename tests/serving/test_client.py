"""The load generator against a fake streaming server that behaves like vLLM's completions endpoint."""

import asyncio
import json
import zlib

import pytest

web = pytest.importorskip("aiohttp.web")

from fastserve.serving.client import run_load  # noqa: E402
from fastserve.serving.workloads import RequestSpec  # noqa: E402


async def _fake_completions(request):
    body = await request.json()
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    usage = {"prompt_tokens": len(body["prompt"])}
    for i in range(1, body["max_tokens"] + 1):
        await asyncio.sleep(0.002)
        chunk = {"choices": [{"text": "x"}], "usage": {**usage, "completion_tokens": i}}
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    final = {"choices": [], "usage": {**usage, "completion_tokens": body["max_tokens"]}}  # usage-only chunk
    final["usage"]["prompt_tokens_details"] = {"cached_tokens": len(body["prompt"]) - 1}  # a prefix-cache hit
    await response.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
    return response


async def _refuse(request):
    return web.Response(status=400, text="bad request")


def _run(load, specs, path=""):
    async def main():
        app = web.Application()
        app.router.add_post("/v1/completions", _fake_completions)
        app.router.add_post("/bad/v1/completions", _refuse)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await run_load(f"http://127.0.0.1:{port}{path}", "fake", specs, load)
        finally:
            await runner.cleanup()

    return asyncio.run(main())


SPECS = [RequestSpec(id=i, prompt=[1, 2, 3], max_tokens=5 + i) for i in range(6)]


@pytest.mark.parametrize("load", [{"mode": "closed", "concurrency": 2}, {"mode": "open", "rate": 200.0}])
def test_every_token_is_timestamped(load):
    results = _run(load, SPECS)
    assert [r.id for r in results] == list(range(6))
    for r, spec in zip(results, SPECS, strict=True):
        assert r.ok and r.output_tokens == spec.max_tokens == sum(r.chunk_tokens)
        assert r.sent < r.first_token <= r.finished
        assert len(r.itls()) == spec.max_tokens - 1
        assert r.cached_tokens == len(spec.prompt) - 1
        assert r.output_crc == zlib.crc32(b"x" * spec.max_tokens)  # the fake server streams "x" per token


def test_http_errors_are_recorded_not_raised():
    results = _run({"mode": "closed", "concurrency": 1}, SPECS[:2], path="/bad")
    assert all(not r.ok and r.error.startswith("HTTP 400") for r in results)
