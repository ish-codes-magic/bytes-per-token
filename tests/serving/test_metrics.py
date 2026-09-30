import pytest

from fastserve.serving.metrics import RequestResult, request_rows, summarize


def _request(i, sent, chunks, error=None):
    """chunks: list of (time, tokens) as a streaming server would deliver them."""
    r = RequestResult(id=i, prompt_len=10, max_tokens=sum(k for _, k in chunks), scheduled=sent, sent=sent)
    r.chunk_times = [t for t, _ in chunks]
    r.chunk_tokens = [k for _, k in chunks]
    r.output_tokens = sum(r.chunk_tokens)
    r.first_token, r.finished = (chunks[0][0], chunks[-1][0]) if chunks else (None, None)
    r.error = error
    return r


def test_per_request_latencies():
    r = _request(0, 1.0, [(1.2, 1), (1.3, 1), (1.5, 2)])  # the last chunk carries 2 tokens
    assert r.ttft == pytest.approx(0.2)
    assert r.e2e == pytest.approx(0.5)
    assert r.tpot == pytest.approx(0.3 / 3)  # (1.5 - 1.2) over 3 more tokens
    assert r.itls() == pytest.approx([0.1, 0.1, 0.1])  # the 2-token chunk contributes 2 equal gaps


def test_summary_throughput_goodput_and_cost():
    results = [
        _request(0, 0.0, [(0.1, 1), (0.2, 1), (0.3, 1)]),  # fast: meets the SLO
        _request(1, 0.0, [(0.9, 1), (1.0, 1)]),  # TTFT 900 ms: misses the 500 ms SLO
        _request(2, 0.0, [], error="HTTP 500"),
    ]
    s = summarize(results, slo_ttft_s=0.5, slo_tpot_s=0.2, dollars_per_hour=0.8)
    assert (s["requests"], s["completed"], s["errors"]) == (3, 2, 1)
    assert s["duration_s"] == pytest.approx(1.0)
    assert s["output_throughput"] == pytest.approx(5.0)  # 5 tokens in 1 s
    assert s["goodput_fraction"] == pytest.approx(0.5)
    assert s["ttft_ms"]["max"] == pytest.approx(900)
    assert s["dollars_per_million_output_tokens"] == pytest.approx(0.8 / (5 * 3600) * 1e6)
    assert request_rows(results)["rows"][1][:3] == [1, 10, 2]


def test_summary_of_nothing():
    assert summarize([], slo_ttft_s=1, slo_tpot_s=1) == {"requests": 0, "completed": 0, "errors": 0}
