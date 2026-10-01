"""Replay greedy speculative decoding exactly, for any draft length, without running it. Stdlib only.

Under greedy decoding the output is fixed: it is the target's own greedy text y, whatever the drafter does.
A draft token at position j is accepted iff it equals y[j], and as long as the draft has matched, the drafter
has seen exactly the target's context. So one teacher-forced pass of the drafter over y gives a bit per
position: `agree[j]` = "the drafter, given the true context, predicts y[j]". From those bits the whole
speculative run follows, for every k:

    a round at position i accepts the run of agreeing positions starting at i (at most k of them),
    then the target supplies one token, and the next round starts after it.

An n-gram drafter needs no model pass at all: its proposal at position i is a lookup in the text so far.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

Rounds = list[tuple[int, int]]  # per round: (tokens proposed, tokens accepted)


def model_rounds(agree: list[bool], k: int) -> Rounds:
    """The rounds of greedy speculation with a model drafter of draft length k, over one output."""
    rounds, i, n = [], 0, len(agree)
    while i < n:
        accepted = 0
        while accepted < k and i + accepted < n and agree[i + accepted]:
            accepted += 1
        rounds.append((k, accepted))
        i += accepted + 1  # the accepted draft tokens, then the target's own token
    return rounds


def ngram_rounds(prompt: list[int], output: list[int], k: int, drafter: Any) -> Rounds:
    """The same for a lookup drafter: `drafter.propose(context, k)` returns (tokens, None)."""
    rounds, i = [], 0
    while i < len(output):
        draft, _ = drafter.propose(prompt + output[:i], k)
        accepted = 0
        while (
            accepted < len(draft) and i + accepted < len(output) and draft[accepted] == output[i + accepted]
        ):
            accepted += 1
        rounds.append((len(draft), accepted))
        i += accepted + 1
    return rounds


def from_draft(rounds: Rounds, n: int) -> list[bool]:
    """Per output token: True if it was an accepted draft token, False if the target supplied it."""
    flags: list[bool] = []
    for _, accepted in rounds:
        flags += [True] * accepted + [False]
    return flags[:n]


def summarize(rounds_per_output: list[Rounds], tokens: int) -> dict[str, Any]:
    """Tokens per target pass, the share of proposed tokens accepted, and the accepted-length histogram."""
    rounds = [r for output in rounds_per_output for r in output]
    proposed = sum(p for p, _ in rounds)
    histogram = Counter(a for _, a in rounds)
    return {
        "rounds": len(rounds),
        "tokens_per_round": tokens / len(rounds) if rounds else 0.0,
        "acceptance_rate": sum(a for _, a in rounds) / proposed if proposed else 0.0,
        "proposed_per_round": proposed / len(rounds) if rounds else 0.0,
        "accepted_histogram": [histogram.get(a, 0) for a in range(max(histogram, default=0) + 1)],
    }


def expected_tokens(alpha: float, k: int) -> float:
    """Tokens per round if every draft token were accepted independently with probability alpha:
    1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α)."""
    return float(k + 1) if alpha >= 1.0 else (1 - alpha ** (k + 1)) / (1 - alpha)


def run_rates(agree: list[bool]) -> dict[str, float]:
    """How often the drafter agrees overall, right after an agreement, and right after a miss.

    If agreements were independent the three would be equal. Real text is "bursty": easy stretches
    (boilerplate, copied spans) agree many times in a row.
    """
    after_hit = [b for a, b in zip(agree, agree[1:], strict=False) if a]
    after_miss = [b for a, b in zip(agree, agree[1:], strict=False) if not a]

    def rate(xs: list[bool]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {"agree": rate(agree), "after_agree": rate(after_hit), "after_miss": rate(after_miss)}
