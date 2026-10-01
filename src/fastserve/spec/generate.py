"""The draft-verify loop: speculative decoding for one sequence, on any target and drafter.

Each round:
1. The drafter proposes up to k tokens after the current context.
2. The target scores the context plus the draft in ONE pass: k + 1 next-token distributions.
3. The rejection sampler accepts a prefix of the draft and adds one token of the target's own.

So a round always yields at least one token (as plain decoding would) and at most k + 1, for one target pass.
In the memory-bound regime that pass costs about the same as generating one token: the weights are streamed
once either way. That is the whole trick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fastserve.engine.sampler import SamplingParams
from fastserve.spec.drafters import Drafter
from fastserve.spec.lm import StepLM
from fastserve.spec.rejection_sampler import verify, warp


@dataclass
class SpecResult:
    tokens: list[int] = field(default_factory=list)
    from_draft: list[bool] = field(
        default_factory=list
    )  # per token: an accepted draft token, or the target's
    rounds: list[tuple[int, int]] = field(default_factory=list)  # per round: (tokens proposed, accepted)

    def tokens_per_round(self) -> float:
        return len(self.tokens) / len(self.rounds) if self.rounds else 0.0


def speculative_generate(
    target: StepLM,
    drafter: Drafter,
    prompt: list[int],
    params: SamplingParams,
    k: int,
    generator: torch.Generator | None = None,
) -> SpecResult:
    """Generate up to `params.max_new_tokens` tokens after `prompt`, verifying k drafted tokens per round."""
    result, context = SpecResult(), list(prompt)
    while len(result.tokens) < params.max_new_tokens:
        draft, q = drafter.propose(context, k, generator)
        # [len(draft) + 1, vocab], on the CPU: sampling uses one CPU generator, whatever device the models use
        p = warp(target.logits_after(context + draft, len(draft) + 1), params).cpu()
        accepted, following = verify(p, q, torch.tensor(draft, dtype=torch.long), generator)
        result.rounds.append((len(draft), accepted))
        for token, drafted in [*((t, True) for t in draft[:accepted]), (following, False)]:
            result.tokens.append(token)
            result.from_draft.append(drafted)
            context.append(token)
            if token in params.stop_token_ids or len(result.tokens) == params.max_new_tokens:
                return result
    return result
