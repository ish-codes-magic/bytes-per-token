import pytest

from fastserve.hw.specs import SPECS, spec_for


@pytest.mark.parametrize(
    ("device_name", "key"),
    [("NVIDIA L4", "L4"), ("Tesla T4", "T4"), ("NVIDIA H100 80GB HBM3", "H100 HBM3")],
)
def test_spec_for_known_devices(device_name, key):
    assert spec_for(device_name) is SPECS[key]


@pytest.mark.parametrize(
    "device_name", ["NVIDIA L40S", "NVIDIA H100 PCIe", "NVIDIA GeForce RTX 3050 Laptop GPU", ""]
)
def test_spec_for_does_not_guess(device_name):
    assert spec_for(device_name) is None


def test_l4_fp8_is_twice_bf16():
    l4 = SPECS["L4"]
    assert l4.peak_flops["fp8"] == 2 * l4.peak_flops["bf16"]  # FP8 tensor cores run at twice the BF16 rate
