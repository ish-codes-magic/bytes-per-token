"""Turning logits into the next token: greedy, temperature and top-p (nucleus) sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0  # 0 = greedy (always the most likely token)
    top_p: float = 1.0  # keep the smallest set of top tokens whose probability reaches top_p
    stop_token_ids: tuple[int, ...] = ()


def sample(
    logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None
) -> torch.Tensor:
    """logits: [B, vocab] → next token ids: [B]."""
    if params.temperature == 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / params.temperature, dim=-1)  # <1 sharpens, >1 flattens
    if params.top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs  # the top token always survives
        sorted_probs[mass_before > params.top_p] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
