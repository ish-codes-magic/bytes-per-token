"""Drafters: cheap guesses at the next k tokens, for the target to verify in one pass.

    propose(context, k) → (tokens, q)
        tokens: up to k proposed token ids
        q:      [len(tokens), vocab] the distribution each token was drawn from, or None when the proposal is
                deterministic (the verifier then treats q as a point mass)

- ModelDrafter runs a smaller language model autoregressively: k cheap steps.
- NGramDrafter (prompt lookup) does no model work at all: it finds the last few tokens earlier in the context
  and proposes whatever followed them. It only helps when the output repeats the input (summaries, code edits,
  quoted text), but there it is nearly free.
"""

from __future__ import annotations

from typing import Protocol

import torch

from fastserve.engine.sampler import SamplingParams
from fastserve.spec.lm import StepLM
from fastserve.spec.rejection_sampler import warp


class Drafter(Protocol):
    def propose(
        self, context: list[int], k: int, generator: torch.Generator | None = None
    ) -> tuple[list[int], torch.Tensor | None]: ...


class ModelDrafter:
    """Draft with a small model: sample k tokens one at a time from its (warped) distribution."""

    def __init__(self, lm: StepLM, params: SamplingParams):
        self.lm, self.params = lm, params

    def propose(
        self, context: list[int], k: int, generator: torch.Generator | None = None
    ) -> tuple[list[int], torch.Tensor | None]:
        tokens: list[int] = []
        rows = []
        for _ in range(k):
            probs = warp(self.lm.logits_after(context + tokens, 1)[0], self.params).cpu()  # [vocab]
            tokens.append(int(torch.multinomial(probs, num_samples=1, generator=generator)))
            rows.append(probs)
        return tokens, torch.stack(rows) if rows else None


class NGramDrafter:
    """Prompt lookup: match the context's last n tokens (longest n first) against earlier text, and propose
    the tokens that followed the most recent match."""

    def __init__(self, max_n: int = 4, min_n: int = 2):
        self.max_n, self.min_n = max_n, min_n

    def propose(
        self, context: list[int], k: int, generator: torch.Generator | None = None
    ) -> tuple[list[int], torch.Tensor | None]:
        for n in range(min(self.max_n, len(context) - 1), self.min_n - 1, -1):
            suffix = context[-n:]
            for start in range(len(context) - n - 1, -1, -1):  # most recent earlier occurrence first
                if context[start : start + n] == suffix:
                    return context[start + n : start + n + k], None
        return [], None


class NoDrafter:
    """Proposes nothing: every round is one target token. Plain decoding, as the k = 0 case of the loop."""

    def propose(
        self, context: list[int], k: int, generator: torch.Generator | None = None
    ) -> tuple[list[int], torch.Tensor | None]:
        return [], None
