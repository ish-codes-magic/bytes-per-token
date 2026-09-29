import pytest

from fastserve.hw.analysis import gpu_spec, measured_bandwidth, measured_peak_flops, metrics_of, single


def test_measured_bandwidth_ignores_transfers_that_fit_in_cache(probe_records):
    # Only 256 MiB and 1 GiB exceed 4 × the 48 MiB L2; the fake read speed at 1 GiB (2^30) is 250 + 30.
    assert measured_bandwidth(probe_records, method="read") == pytest.approx(280e9)
    assert measured_bandwidth(probe_records, method="copy") == pytest.approx(230e9)


def test_measured_peak_flops_skips_errors(probe_records):
    peaks = measured_peak_flops(probe_records)
    assert peaks["bf16"] == pytest.approx(90e12)
    assert peaks["int8"] == pytest.approx(150e12)  # the int8 M=1 rows are errors, not zeros
    errors = [m for m in metrics_of(probe_records, "matmul") if "error" in m]
    assert errors and all(m["format"] == "int8" for m in errors)


def test_gpu_spec_and_single(probe_records):
    assert gpu_spec(probe_records)["name"] == "NVIDIA L4"
    assert single(probe_records, "launch_overhead")["eager_us_per_kernel"] == 6.0
    assert single(probe_records, "does_not_exist") is None
