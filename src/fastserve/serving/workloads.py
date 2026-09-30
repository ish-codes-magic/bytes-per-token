"""Workloads: what the requests look like, and when they arrive. Stdlib only, fully seeded.

Prompts are random token ids (sent to the server as ids, so there's no tokenizer in the loop). Speed doesn't
depend on the words, only on the lengths, and on shared prefixes, which the agent workload models explicitly.
Outputs have a forced length (`ignore_eos`), so every configuration does exactly the same work.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

# Qwen3's ordinary tokens are ids 0 … 151642; ids from 151643 up are special (end of text, chat markers).
DEFAULT_VOCAB = 151_643


@dataclass(frozen=True)
class LengthDist:
    """A token-length distribution: fixed, uniform, or log-normal (the long tail of real chat traffic)."""

    kind: str
    value: int = 0
    low: int = 1
    high: int = 1
    median: float = 0.0
    sigma: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | int) -> LengthDist:
        return cls(kind="fixed", value=cfg) if isinstance(cfg, int) else cls(**cfg)

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return self.value
        if self.kind == "uniform":
            return rng.randint(self.low, self.high)
        if self.kind == "lognormal":  # median = e^μ; clipped to [low, high]
            return min(self.high, max(self.low, round(rng.lognormvariate(math.log(self.median), self.sigma))))
        raise ValueError(f"unknown length distribution {self.kind!r}")


@dataclass(frozen=True)
class RequestSpec:
    id: int
    prompt: list[int]  # token ids
    max_tokens: int  # output length (forced)


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: LengthDist
    output_len: LengthDist
    num_requests: int
    shared_prefix_len: int = 0  # tokens every request starts with (a system prompt, tool descriptions)
    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> Workload:
        return cls(
            name=name,
            input_len=LengthDist.from_config(cfg["input_len"]),
            output_len=LengthDist.from_config(cfg["output_len"]),
            num_requests=cfg["num_requests"],
            shared_prefix_len=cfg.get("shared_prefix_len", 0),
            seed=cfg.get("seed", 0),
        )

    def requests(self) -> list[RequestSpec]:
        """The same list every time (seeded): every load point and configuration sees identical requests."""
        rng = random.Random(self.seed)
        prefix = [rng.randrange(self.vocab_size) for _ in range(self.shared_prefix_len)]
        specs = []
        for i in range(self.num_requests):
            suffix = [rng.randrange(self.vocab_size) for _ in range(self.input_len.sample(rng))]
            specs.append(RequestSpec(id=i, prompt=prefix + suffix, max_tokens=self.output_len.sample(rng)))
        return specs


def poisson_arrivals(rate_per_s: float, n: int, seed: int = 0) -> list[float]:
    """Open-loop arrival times (seconds from the start): exponential gaps with mean 1/rate."""
    rng = random.Random(seed)
    t, times = 0.0, []
    for _ in range(n):
        times.append(t)
        t += rng.expovariate(rate_per_s)
    return times
