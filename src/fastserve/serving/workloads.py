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
    max_tokens: int  # output length: forced when ignore_eos, otherwise a cap
    ignore_eos: bool = (
        True  # random-token workloads force the length; real prompts stop where the model stops
    )
    task: str = ""  # for real prompts: which kind of request this is


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: LengthDist
    output_len: LengthDist
    num_requests: int
    shared_prefix_len: int = 0  # tokens every request starts with (a system prompt, tool descriptions)
    prefixes: int = 1  # distinct shared prefixes (apps), each `shared_prefix_len` long
    turns: int = 1  # turns per conversation: each turn's prompt extends the previous turn's
    reply_len: int = 0  # tokens standing in for the assistant's previous answer between turns
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
            prefixes=cfg.get("prefixes", 1),
            turns=cfg.get("turns", 1),
            reply_len=cfg.get("reply_len", 0),
            seed=cfg.get("seed", 0),
        )

    def requests(self) -> list[RequestSpec]:
        """The same list every time (seeded): every load point and configuration sees identical requests."""
        if self.turns > 1 or self.prefixes > 1:
            return self._conversations()
        rng = random.Random(self.seed)
        prefix = [rng.randrange(self.vocab_size) for _ in range(self.shared_prefix_len)]
        specs = []
        for i in range(self.num_requests):
            suffix = [rng.randrange(self.vocab_size) for _ in range(self.input_len.sample(rng))]
            specs.append(RequestSpec(id=i, prompt=prefix + suffix, max_tokens=self.output_len.sample(rng)))
        return specs

    def _conversations(self) -> list[RequestSpec]:
        """Multi-turn chats across several apps: the structure a radix-tree prefix cache is built for.

        Conversation c belongs to app c mod `prefixes` and starts with that app's system prompt. Turn t's
        prompt is turn t−1's prompt + a stand-in reply (`reply_len` tokens) + a new user message. Requests
        are ordered turn by turn (every conversation's first turn, then every second turn, ...), so a turn's
        predecessor has usually finished, as in real chat traffic.
        """
        rng = random.Random(self.seed)

        def tokens(n: int) -> list[int]:
            return [rng.randrange(self.vocab_size) for _ in range(n)]

        systems = [tokens(self.shared_prefix_len) for _ in range(self.prefixes)]
        conversations = self.num_requests // self.turns
        history = [list(systems[c % self.prefixes]) for c in range(conversations)]
        specs = []
        for turn in range(self.turns):
            for c in range(conversations):
                if turn:
                    history[c] += tokens(self.reply_len)
                history[c] += tokens(self.input_len.sample(rng))
                specs.append(
                    RequestSpec(
                        id=len(specs), prompt=list(history[c]), max_tokens=self.output_len.sample(rng)
                    )
                )
        return specs


def text_requests(prompts: dict[str, list[list[int]]], max_tokens: int) -> list[RequestSpec]:
    """Real prompts (token ids per task) as requests, interleaved task by task so any prefix of the list is
    a balanced mix. Outputs end where the model ends them (or at `max_tokens`)."""
    specs: list[RequestSpec] = []
    for i in range(max(len(p) for p in prompts.values())):
        for task, task_prompts in prompts.items():
            if i < len(task_prompts):
                specs.append(
                    RequestSpec(
                        id=len(specs),
                        prompt=task_prompts[i],
                        max_tokens=max_tokens,
                        ignore_eos=False,
                        task=task,
                    )
                )
    return specs


def poisson_arrivals(rate_per_s: float, n: int, seed: int = 0) -> list[float]:
    """Open-loop arrival times (seconds from the start): exponential gaps with mean 1/rate."""
    rng = random.Random(seed)
    t, times = 0.0, []
    for _ in range(n):
        times.append(t)
        t += rng.expovariate(rate_per_s)
    return times
