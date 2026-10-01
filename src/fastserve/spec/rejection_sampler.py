"""Speculative sampling: accept or reject drafted tokens so the output follows the *target* exactly.

A drafter proposes tokens x₁ … x_k, with x_i drawn from its distribution q_i. The target model then gives its
own distribution p_i for each of those positions in one forward pass. Going left to right:

    accept x_i with probability min(1, p_i(x_i) / q_i(x_i))
    on the first rejection, draw a replacement from  norm(max(0, p_i − q_i))  and stop
    if all k are accepted, draw one more token from p_{k+1} (the pass computed it anyway)

Why the output is distributed exactly as p_i (Leviathan et al. 2023; Chen et al. 2023). For any token x:

    P(output = x) = P(draft x and accept it) + P(reject) · P(replacement = x)
                  = q(x)·min(1, p(x)/q(x)) + R · max(0, p(x) − q(x)) / Z
                  = min(p(x), q(x)) + max(0, p(x) − q(x))
                  = p(x)

    where R = P(reject) = 1 − Σ_y min(p(y), q(y)) and Z = Σ_y max(0, p(y) − q(y)) are the same number: the
    probability mass where p exceeds q equals the mass where q exceeds p, so R / Z = 1.

So speculation changes *when* tokens are computed, never *which* tokens are likely: it is lossless.

Two special cases fall out of the same rule:
- Greedy decoding: p and q are one-hot. A draft token is accepted iff it is the target's argmax.
- A deterministic drafter (n-gram lookup): q is one-hot on the proposed token, so it is accepted with
  probability p(x), and the replacement comes from p with x removed.

Shapes: p [k + 1, vocab], q [k, vocab], draft [k].
"""

from __future__ import annotations

import torch

from fastserve.engine.sampler import SamplingParams


def warp(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Logits [..., vocab] → the distribution actually sampled from (float32): temperature, then top-p.

    Temperature 0 gives a one-hot on the argmax, so greedy decoding is the same rule with a point mass.
    """
    if params.temperature == 0.0:
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()
    probs = torch.softmax(logits.float() / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs  # the top token always survives
        sorted_probs = sorted_probs.masked_fill(mass_before > params.top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def _draw(probs: torch.Tensor, generator: torch.Generator | None) -> int:
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


def verify(
    p: torch.Tensor,
    q: torch.Tensor | None,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Check `draft` against the target. Returns (number of draft tokens accepted, the token that follows).

    p: [k + 1, vocab] target distributions for the k drafted positions and the one after.
    q: [k, vocab] the drafter's distributions, or None for a deterministic drafter (one-hot on each token).
    The token that follows is the replacement at the first rejection, or a fresh draw from p[k] if every
    draft token was accepted. Either way the caller gains `accepted + 1` tokens from one target pass.
    """
    k = len(draft)
    uniform = torch.rand(k, generator=generator) if k else None
    for i in range(k):
        x = int(draft[i])
        q_x = 1.0 if q is None else float(q[i, x])
        if float(uniform[i]) * q_x < float(p[i, x]):  # u < p/q, written without dividing by a tiny q
            continue
        if q is None:
            residual = p[i].clone()
            residual[x] = 0.0
        else:
            residual = (p[i] - q[i]).clamp_min(0.0)
        if (
            float(residual.sum()) <= 0.0
        ):  # p == q up to rounding: rejection has probability ~0; fall back to p
            residual = p[i]
        return i, _draw(residual, generator)
    return k, _draw(p[k], generator)
