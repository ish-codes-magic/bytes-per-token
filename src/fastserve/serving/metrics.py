"""Serving metrics from per-request timestamps. Stdlib only.

For one request (times in seconds, measured by the client):
    TTFT  = first token arrives − request sent          ("how long until it starts typing")
    TPOT  = (last token − first token) / (tokens − 1)   ("how fast it types")
    ITL   = gaps between consecutive tokens (a distribution, not one number)
    E2E   = last token − request sent
Across a run: throughput (requests/s, output tokens/s), goodput (only requests that met the SLO), cost.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from fastserve.timing import percentile


@dataclass
class RequestResult:
    id: int
    prompt_len: int
    max_tokens: int
    scheduled: float  # intended send time (open loop) or actual send time (closed loop)
    sent: float = 0.0
    first_token: float | None = None
    finished: float | None = None
    output_tokens: int = 0
    chunk_times: list[float] = field(default_factory=list)  # when each streamed chunk arrived
    chunk_tokens: list[int] = field(default_factory=list)  # how many tokens each chunk carried
    cached_tokens: int | None = None  # prompt tokens served from the prefix cache (if the server reports it)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.first_token is not None and self.finished is not None

    @property
    def ttft(self) -> float:
        return self.first_token - self.sent

    @property
    def e2e(self) -> float:
        return self.finished - self.sent

    @property
    def tpot(self) -> float | None:
        return (
            (self.finished - self.first_token) / (self.output_tokens - 1) if self.output_tokens > 1 else None
        )

    def itls(self) -> list[float]:
        """Per-token gaps. A chunk carrying k tokens contributes k equal gaps (a chunk can hold several)."""
        gaps = []
        for prev, now, k in zip(self.chunk_times, self.chunk_times[1:], self.chunk_tokens[1:], strict=False):
            gaps.extend([(now - prev) / k] * k)
        return gaps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dist(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize(
    results: list[RequestResult],
    *,
    slo_ttft_s: float,
    slo_tpot_s: float,
    dollars_per_hour: float | None = None,
    itl_sample: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate one load point. Latencies in milliseconds in the output, rates per second."""
    ok = [r for r in results if r.ok]
    summary: dict[str, Any] = {
        "requests": len(results),
        "completed": len(ok),
        "errors": len(results) - len(ok),
    }
    if not ok:
        return summary
    duration = max(r.finished for r in ok) - min(r.sent for r in ok)
    output_tokens = sum(r.output_tokens for r in ok)
    good = [r for r in ok if r.ttft <= slo_ttft_s and (r.tpot is None or r.tpot <= slo_tpot_s)]
    itls = [gap for r in ok for gap in r.itls()]
    ms = 1e3
    summary.update(
        duration_s=duration,
        request_throughput=len(ok) / duration,
        output_throughput=output_tokens / duration,
        total_throughput=(output_tokens + sum(r.prompt_len for r in ok)) / duration,
        goodput_requests=len(good) / duration,
        goodput_fraction=len(good) / len(ok),
        slo={"ttft_ms": slo_ttft_s * ms, "tpot_ms": slo_tpot_s * ms},
        ttft_ms=_dist([r.ttft * ms for r in ok]),
        tpot_ms=_dist([r.tpot * ms for r in ok if r.tpot is not None]),
        itl_ms=_dist([g * ms for g in itls]),
        e2e_ms=_dist([r.e2e * ms for r in ok]),
        itl_sample_ms=[g * ms for g in random.Random(seed).sample(itls, min(itl_sample, len(itls)))],
    )
    if dollars_per_hour is not None:
        summary["dollars_per_million_output_tokens"] = (
            dollars_per_hour / (summary["output_throughput"] * 3600) * 1e6
        )
    return summary


def request_rows(results: list[RequestResult]) -> dict[str, Any]:
    """Compact per-request timing table for CDFs and swimlane plots (a columnar dict keeps files small)."""
    columns = [
        "id",
        "prompt_len",
        "output_tokens",
        "scheduled",
        "sent",
        "first_token",
        "finished",
        "cached_tokens",
    ]
    return {"columns": columns, "rows": [[getattr(r, c) for c in columns] for r in results if r.ok]}
