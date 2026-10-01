"""The one thing speculative decoding asks of a language model: logits for the last few context positions.

    logits_after(context, last) → [last, vocab]
        row i predicts the token that follows context[len(context) − last + i]

Verifying k drafted tokens is `logits_after(context + draft, k + 1)`: one call, k + 1 distributions. Two
implementations share the draft-verify loop:

- CachedLM wraps a nanoserve model and its KV cache. It remembers which tokens the cache holds, so each call
  only runs the tokens that are new. When drafted tokens are rejected, the next call's context differs from
  what the cache holds; the cache is "rolled back" simply by overwriting from the first differing position
  (slots past the current position are invisible to the causal mask until they are rewritten).
- TableLM is a toy bigram model: the next-token distribution depends only on the last token. Its exact output
  distribution is known in closed form, which is what the losslessness test needs.
"""

from __future__ import annotations

from typing import Protocol

import torch

from fastserve.engine.kv_cache import ContiguousKVCache
from fastserve.engine.model import CausalLM


class StepLM(Protocol):
    def logits_after(self, context: list[int], last: int) -> torch.Tensor:
        """[last, vocab]: logits for the token following each of the final `last` context positions."""


class TableLM:
    """A bigram "model": row t of `table` is the distribution of the token after t. [vocab, vocab]."""

    def __init__(self, table: torch.Tensor):
        self.log_table = table.clamp_min(1e-30).log()

    def logits_after(self, context: list[int], last: int) -> torch.Tensor:
        return self.log_table[torch.tensor(context[-last:])]


class CachedLM:
    """A nanoserve model with a KV cache that follows the context across calls, including rollbacks."""

    def __init__(self, model: CausalLM, max_len: int, chunk: int = 512):
        weight = model.lm_head.weight
        self.model, self.chunk, self.device = model, chunk, weight.device
        self.cache = ContiguousKVCache(
            model.config, max_batch=1, max_len=max_len, dtype=weight.dtype, device=weight.device
        )
        self.held: list[int] = []  # the tokens whose keys/values the cache holds, in position order
        self.tokens_run = 0  # forward-pass tokens so far: the work actually done

    @torch.inference_mode()
    def logits_after(self, context: list[int], last: int) -> torch.Tensor:
        keep = 0  # how much of the cache still matches this context
        for held, wanted in zip(self.held, context, strict=False):
            if held != wanted:
                break
            keep += 1
        start = min(keep, len(context) - last)  # the last `last` positions must be run to get their logits
        rows = []
        for begin in range(start, len(context), self.chunk):  # chunked, so a long prompt fits in memory
            piece = context[begin : begin + self.chunk]
            ids = torch.tensor(piece, device=self.device)[None]  # [1, Q]
            positions = torch.arange(begin, begin + len(piece), device=self.device)[None]
            rows.append(self.model(ids, positions, self.cache)[0])  # [Q, vocab]
            self.tokens_run += len(piece)
        self.held = list(context)
        return torch.cat(rows)[-last:]
