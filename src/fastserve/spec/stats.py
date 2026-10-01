"""Statistics for the losslessness test: does a sample follow the distribution it should? Stdlib only.

- `chi_square(counts, probs)`: Pearson's statistic against a known distribution, with rare outcomes pooled so
  every bin expects at least `min_expected` samples (the usual condition for the chi-square approximation).
- `chi_square_limit(dof, sigmas)`: a rejection threshold. A chi-square variable with d degrees of freedom has
  mean d and standard deviation √(2d); a correct sampler stays below d + 5·√(2d) essentially always.
- `total_variation(a, b)`: half the L1 distance between two empirical distributions: the largest difference in
  probability they assign to any set of outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping


def chi_square(
    counts: Mapping[Hashable, int], probs: Mapping[Hashable, float], min_expected: float = 5.0
) -> tuple[float, int]:
    """(statistic, degrees of freedom) of observed `counts` against expected `probs` (which sum to 1)."""
    n = sum(counts.values())
    statistic, bins, pooled_observed, pooled_expected = 0.0, 0, 0.0, 0.0
    for outcome, prob in probs.items():
        expected, observed = n * prob, counts.get(outcome, 0)
        if expected < min_expected:
            pooled_observed, pooled_expected = pooled_observed + observed, pooled_expected + expected
            continue
        statistic += (observed - expected) ** 2 / expected
        bins += 1
    pooled_observed += sum(
        c for outcome, c in counts.items() if outcome not in probs
    )  # "impossible" outcomes
    if pooled_expected > 0 or pooled_observed > 0:
        statistic += (pooled_observed - pooled_expected) ** 2 / max(pooled_expected, 1e-9)
        bins += 1
    return statistic, max(bins - 1, 1)


def chi_square_limit(dof: int, sigmas: float = 5.0) -> float:
    return dof + sigmas * math.sqrt(2 * dof)


def total_variation(a: Mapping[Hashable, float], b: Mapping[Hashable, float]) -> float:
    """Half the L1 distance between two distributions given as counts or probabilities (each normalized)."""
    total_a, total_b = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / total_a - b.get(k, 0) / total_b) for k in set(a) | set(b))
