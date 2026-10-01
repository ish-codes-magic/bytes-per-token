import random
import statistics

import pytest

from fastserve.serving.workloads import LengthDist, Workload, poisson_arrivals, text_requests


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


def test_conversations_extend_each_turn_and_group_by_app():
    cfg = {
        "shared_prefix_len": 30,
        "prefixes": 2,
        "turns": 3,
        "reply_len": 7,
        "input_len": 5,
        "output_len": 4,
        "num_requests": 12,  # 4 conversations × 3 turns
    }
    specs = Workload.from_config("multi_turn", cfg).requests()
    assert [len(s.prompt) for s in specs] == [35] * 4 + [47] * 4 + [59] * 4  # + reply 7 + message 5 per turn
    first, second = specs[:4], specs[4:8]
    for a, b in zip(first, second, strict=True):
        assert b.prompt[: len(a.prompt)] == a.prompt  # each turn extends the conversation's previous prompt
    apps = [tuple(s.prompt[:30]) for s in first]
    assert apps[0] == apps[2] != apps[1] == apps[3]  # conversations alternate between the two apps
    assert Workload.from_config("multi_turn", cfg).requests() == specs


def test_single_prefix_workloads_are_unchanged_by_the_conversation_fields():
    cfg = {"input_len": 20, "output_len": 5, "num_requests": 3, "shared_prefix_len": 10}
    plain = Workload.from_config("agent", cfg).requests()
    assert plain == Workload.from_config("agent", {**cfg, "prefixes": 1, "turns": 1}).requests()


def test_text_requests_interleave_tasks_and_let_the_model_stop():
    specs = text_requests({"chat": [[1], [2], [3]], "code": [[7, 7], [8, 8]]}, max_tokens=64)
    assert [s.task for s in specs] == ["chat", "code", "chat", "code", "chat"]
    assert [s.prompt for s in specs] == [[1], [7, 7], [2], [8, 8], [3]]
    assert [s.id for s in specs] == list(range(5))
    assert all(s.max_tokens == 64 and not s.ignore_eos for s in specs)
