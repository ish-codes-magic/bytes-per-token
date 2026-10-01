import random
from collections import Counter

import pytest

from fastserve.spec.stats import chi_square, chi_square_limit, total_variation


def test_chi_square_accepts_a_fair_sample_and_rejects_a_biased_one():
    rng = random.Random(0)
    probs = {"a": 0.5, "b": 0.3, "c": 0.2}
    fair = Counter(rng.choices(list(probs), weights=probs.values(), k=5000))
    statistic, dof = chi_square(fair, probs)
    assert dof == 2 and statistic < chi_square_limit(dof)
    biased = Counter(rng.choices(list(probs), weights=[0.4, 0.4, 0.2], k=5000))
    assert chi_square(biased, probs)[0] > chi_square_limit(2)


def test_rare_and_impossible_outcomes_are_pooled():
    probs = {"common": 0.98, "rare1": 0.01, "rare2": 0.01}
    statistic, dof = chi_square({"common": 98, "rare1": 1, "rare2": 1}, probs)  # n = 100: rares expect 1 each
    assert dof == 1 and statistic == pytest.approx(0.0)
    statistic, _ = chi_square({"common": 90, "never": 10}, probs)  # an outcome the target gives probability 0
    assert statistic > chi_square_limit(1)


def test_total_variation():
    assert total_variation({"a": 1, "b": 1}, {"a": 50, "b": 50}) == 0.0
    assert total_variation({"a": 1}, {"b": 1}) == 1.0
    assert total_variation({"a": 3, "b": 1}, {"a": 1, "b": 1}) == pytest.approx(0.25)
