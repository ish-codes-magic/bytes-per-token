"""FP8 and NF4: our from-scratch rounding against PyTorch's float8 casts and bitsandbytes' NF4 table."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.quant.formats import (  # noqa: E402
    E4M3,
    E5M2,
    fp8_fake_quantize,
    nf4_fake_quantize,
    nf4_levels,
    representable_values,
    round_to_format,
)
from fastserve.quant.rtn import IntSpec, fake_quantize  # noqa: E402

# bitsandbytes' NF4 code (functional.py), which QLoRA published
BITSANDBYTES_NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


@pytest.mark.parametrize(("fmt", "dtype"), [(E4M3, "float8_e4m3fn"), (E5M2, "float8_e5m2")])
def test_fp8_rounding_matches_pytorch(fmt, dtype):
    torch_dtype = getattr(torch, dtype)
    g = torch.Generator().manual_seed(0)
    x = torch.cat(
        [
            torch.randn(4000, generator=g)
            * 10.0 ** torch.randint(-6, 3, (4000,), generator=g),  # many scales
            representable_values(fmt),  # exact grid points
            (representable_values(fmt)[1:] + representable_values(fmt)[:-1]) / 2,  # exact ties
        ]
    )
    x = x.clamp(-fmt.max_value, fmt.max_value)
    assert torch.equal(round_to_format(x, fmt), x.to(torch_dtype).float())


def test_fp8_saturates_and_counts_its_values():
    assert round_to_format(torch.tensor([1000.0, -1e6]), E4M3).tolist() == [448.0, -448.0]
    assert len(representable_values(E4M3)) == 253  # 126 positive, 126 negative, one zero
    assert len(representable_values(E5M2)) == 247


def test_fp8_scaling_maps_each_rows_max_onto_the_format_max():
    w = torch.randn(16, 64, generator=torch.Generator().manual_seed(1))
    w_hat = fp8_fake_quantize(w, E4M3, "channel")
    assert torch.allclose(w_hat.abs().amax(dim=-1), w.abs().amax(dim=-1), rtol=1e-6)
    rel = ((w_hat - w).abs() / w.abs().amax(dim=-1, keepdim=True)).max()
    assert rel < 2**-4  # 3 mantissa bits: at most half a step of 1/8 relative to the row max


def test_nf4_levels_are_normal_quantiles_matching_bitsandbytes():
    levels = nf4_levels()
    assert len(levels) == 16 and levels[0] == -1.0 and levels[-1] == 1.0 and 0.0 in levels
    assert levels == pytest.approx(BITSANDBYTES_NF4, abs=1e-5)


def test_nf4_beats_uniform_int4_on_bell_shaped_weights():
    w = torch.randn(256, 1024, generator=torch.Generator().manual_seed(2)) * 0.02
    nf4 = (nf4_fake_quantize(w, 64) - w).pow(2).mean()
    int4 = (fake_quantize(w, IntSpec(4, "group", 64)) - w).pow(2).mean()
    assert nf4 < int4
    assert len(set(nf4_fake_quantize(w, 64)[0, :64].tolist())) <= 16
