import pytest

from fastserve.hw.analysis import (
    gpu_spec,
    m0_observables,
    measured_bandwidth,
    measured_peak_flops,
    metrics_of,
    single,
)


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


def test_m0_observables_on_fake_run(probe_records):
    obs = m0_observables(probe_records)
    assert obs["read_bandwidth_gbps"] == pytest.approx(280)
    assert obs["copy_bandwidth_gbps"] == pytest.approx(230)
    assert obs["l2_speedup"] == pytest.approx(274 / 280)  # best 1-32 MiB transfer (read, 16 MiB) vs memory
    assert obs["tiny_transfer_gbps"] == pytest.approx(262)
    assert obs["bf16_peak_tflops"] == pytest.approx(90)
    assert obs["fp8_over_bf16"] == pytest.approx(170 / 90)
    assert obs["bf16_m1_pct_of_peak"] is None  # the fake run has no N=K=4096 shape: missing, not zero
    assert obs["bf16_ridge"] == pytest.approx(90e12 / 280e9)
    assert obs["copy_power_w"] is None
    assert obs["matmul_power_w"] == 70.0
