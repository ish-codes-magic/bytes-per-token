import pytest

from fastserve.perfmodel.roofline import (
    arithmetic_intensity,
    attainable_flops,
    matmul_bytes,
    matmul_flops,
    matmul_intensity,
    ridge_point,
)

L4_BF16, L4_BW = 121e12, 300e9


def test_ridge_point_of_l4_datasheet():
    assert ridge_point(peak_flops=L4_BF16, bandwidth=L4_BW) == pytest.approx(403.3, abs=0.1)


def test_attainable_flops_has_two_regimes():
    ridge = ridge_point(peak_flops=L4_BF16, bandwidth=L4_BW)
    assert attainable_flops(1.0, peak_flops=L4_BF16, bandwidth=L4_BW) == L4_BW  # memory-bound: slope
    assert attainable_flops(10 * ridge, peak_flops=L4_BF16, bandwidth=L4_BW) == L4_BF16  # compute-bound: roof


def test_matmul_counts():
    assert matmul_flops(2, 3, 4) == 48
    assert matmul_bytes(2, 3, 4, a_bytes=2, b_bytes=2, out_bytes=2) == 2 * (8 + 12 + 6)


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_decode_intensity_is_about_the_batch_size(batch):
    # A decode step multiplies `batch` token vectors by a 4096 x 4096 BF16 weight: AI ≈ batch.
    ai = matmul_intensity(batch, 4096, 4096, a_bytes=2, b_bytes=2, out_bytes=2)
    assert ai == pytest.approx(batch, rel=0.04)


def test_arithmetic_intensity_rejects_zero_bytes():
    with pytest.raises(ValueError):
        arithmetic_intensity(1.0, 0.0)
