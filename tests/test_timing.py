import pytest

from fastserve.timing import percentile, summarize


def test_percentile_interpolates_like_numpy():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0
    assert percentile(values, 90) == pytest.approx(4.6)  # numpy.percentile(values, 90) == 4.6


def test_percentile_ignores_input_order():
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0


@pytest.mark.parametrize("bad", [-1, 101])
def test_percentile_rejects_out_of_range_q(bad):
    with pytest.raises(ValueError):
        percentile([1.0], bad)


def test_summarize_reports_median_and_tail():
    stats = summarize([10.0, 11.0, 12.0, 13.0, 100.0])  # one slow outlier
    assert stats.n == 5
    assert stats.median_ms == 12.0  # the median shrugs off the outlier ...
    assert stats.mean_ms == pytest.approx(29.2)  # ... the mean does not
    assert stats.min_ms == 10.0
    assert stats.p99_ms > stats.p90_ms > stats.median_ms


def test_summarize_single_value_has_zero_spread():
    assert summarize([4.0]).stdev_ms == 0.0


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])
