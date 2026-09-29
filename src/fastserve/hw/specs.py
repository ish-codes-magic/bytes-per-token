"""Datasheet numbers for the GPUs this project meets, used only to compare *measured* against *promised*.

All compute numbers are dense (no 2:4 sparsity). Datasheets list sparse numbers prominently; they are 2× the
dense ones and never apply to our workloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MIB = 2**20


@dataclass(frozen=True)
class GpuSpec:
    name: str
    bandwidth: float  # bytes/s
    peak_flops: dict[str, float]  # number format -> dense FLOP/s (OP/s for int8)
    l2_bytes: int
    power_w: float

    def to_dict(self) -> dict:
        return asdict(self)


# Keys are the words that identify each GPU in torch.cuda.get_device_name() (e.g. "NVIDIA L4",
# "Tesla T4", "NVIDIA H100 80GB HBM3" for the SXM part).
SPECS: dict[str, GpuSpec] = {
    "L4": GpuSpec(
        name="NVIDIA L4",
        bandwidth=300e9,
        peak_flops={"bf16": 121e12, "fp16": 121e12, "fp8": 242e12, "int8": 242e12},
        l2_bytes=48 * MIB,
        power_w=72,
    ),
    "T4": GpuSpec(
        name="NVIDIA T4",
        bandwidth=320e9,
        peak_flops={"fp16": 65e12, "int8": 130e12},
        l2_bytes=4 * MIB,
        power_w=70,
    ),
    "H100 HBM3": GpuSpec(
        name="NVIDIA H100 SXM",
        bandwidth=3.35e12,
        peak_flops={"bf16": 989e12, "fp16": 989e12, "fp8": 1979e12, "int8": 1979e12},
        l2_bytes=50 * MIB,
        power_w=700,
    ),
}


def spec_for(device_name: str) -> GpuSpec | None:
    """Look up a spec from a device name such as torch's "NVIDIA L4". None if the GPU is unknown.

    Matches whole words, so "NVIDIA L40S" is not mistaken for an L4.
    """
    words = set(device_name.upper().split())
    for key, spec in SPECS.items():
        if set(key.upper().split()) <= words:
            return spec
    return None
