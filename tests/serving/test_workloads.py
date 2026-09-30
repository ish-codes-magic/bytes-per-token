import random
import statistics

import pytest

from fastserve.serving.workloads import LengthDist, Workload, poisson_arrivals


def test_length_distributions_respect_their_bounds():
    rng = random.Random(0)
    assert LengthDist.from_config(128).sample(rng) == 128
    uniform = LengthDist(kind="uniform", low=10, high=20)
    assert all(10 <= uniform.sample(rng) <= 20 for _ in range(500))
    lognormal = LengthDist(kind="lognormal", median=200, sigma=1.0, low=16, high=2048)
    draws = [lognormal.sample(rng) for _ in range(5000)]
    assert all(16 <= d <= 2048 for d in draws)
    assert statistics.median(draws) == pytest.approx(200, rel=0.1)
    assert max(draws) > 3 * statistics.median(draws)  # the long tail of real chat traffic


def test_workload_is_reproducible_and_shares_its_prefix():
    cfg = {"input_len": 20, "output_len": {"kind": "uniform", "low": 5, "high": 9}, "num_requests": 4}
    workload = Workload.from_config("agent", {**cfg, "shared_prefix_len": 50, "seed": 3})
    a, b = workload.requests(), workload.requests()
    assert a == b
    assert all(len(r.prompt) == 70 for r in a)
    assert len({tuple(r.prompt[:50]) for r in a}) == 1  # identical prefix...
    assert len({tuple(r.prompt[50:]) for r in a}) == 4  # ...different suffixes
    assert all(max(r.prompt) < workload.vocab_size for r in a)  # never a special token


def test_poisson_arrivals_have_the_requested_rate():
    times = poisson_arrivals(rate_per_s=8.0, n=5000, seed=1)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert times[0] == 0.0 and all(g >= 0 for g in gaps)
    assert statistics.fmean(gaps) == pytest.approx(1 / 8, rel=0.05)
    assert statistics.stdev(gaps) == pytest.approx(1 / 8, rel=0.1)  # exponential: std = mean
