"""Perplexity, and KL divergence against a reference model, on fixed windows of text (teacher forcing).

Teacher forcing: every position is predicted from the *true* previous tokens, so the reference and the
candidate see identical inputs and their next-token distributions can be compared directly:

    KL(P_ref ‖ P_cand) = Σ_x P_ref(x) · (log P_ref(x) − log P_cand(x))     per position, then averaged
    perplexity         = exp(mean negative log-likelihood of the true next token)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

LogitsFn = Callable[[torch.Tensor], torch.Tensor]  # [batch, seq] token ids -> [batch, seq, vocab] logits


def token_windows(ids: list[int], window: int, max_windows: int) -> torch.Tensor:
    """Cut a long token stream into non-overlapping windows: [n_windows, window]."""
    n = min(max_windows, len(ids) // window)
    if n == 0:
        raise ValueError(f"need at least {window} tokens, got {len(ids)}")
    return torch.tensor(ids[: n * window]).view(n, window)


@torch.inference_mode()
def evaluate(
    windows: torch.Tensor, reference: LogitsFn, candidate: LogitsFn | None = None, *, batch: int = 4
) -> dict[str, Any]:
    """Reference (and candidate) perplexity; with a candidate, also KL(ref ‖ cand) and top-1 agreement."""
    totals = {"nll_ref": 0.0, "nll_cand": 0.0, "kl": 0.0, "agree": 0.0, "positions": 0}
    for start in range(0, len(windows), batch):
        ids = windows[start : start + batch]
        targets = ids[:, 1:]  # predict token t+1 from tokens ≤ t
        ref = torch.log_softmax(reference(ids)[:, :-1].float(), dim=-1)  # [B, T-1, vocab]
        totals["nll_ref"] -= ref.gather(-1, targets.to(ref.device)[..., None]).sum().item()
        totals["positions"] += targets.numel()
        if candidate is not None:
            cand = torch.log_softmax(candidate(ids)[:, :-1].float(), dim=-1)
            totals["nll_cand"] -= cand.gather(-1, targets.to(cand.device)[..., None]).sum().item()
            totals["kl"] += (ref.exp() * (ref - cand)).sum().item()
            totals["agree"] += (ref.argmax(-1) == cand.argmax(-1)).sum().item()
    n = totals["positions"]
    result: dict[str, Any] = {"positions": n, "perplexity_ref": math.exp(totals["nll_ref"] / n)}
    if candidate is not None:
        result.update(
            perplexity_cand=math.exp(totals["nll_cand"] / n),
            mean_kl=totals["kl"] / n,
            top1_agreement=totals["agree"] / n,
        )
    return result
