"""GPTQ against its definition: a naive column-by-column OBS loop, RTN when inputs are uncorrelated, and a
lower layer-output error than RTN when they are correlated."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.quant.gptq import GPTQConfig, HessianAccumulator, gptq_quantize, layer_loss  # noqa: E402
from fastserve.quant.rtn import IntSpec, fake_quantize, grouped, round_to_grid, scale_and_zero  # noqa: E402


def correlated_problem(rows=32, cols=64, tokens=2048, seed=0):
    """Weights, and a Hessian from inputs whose features are mixed together (so they correlate)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g)
    mixing = torch.randn(cols, cols, generator=g) / cols**0.5 + torch.eye(cols)
    x = torch.randn(tokens, cols, generator=g) @ mixing  # [tokens, in]
    acc = HessianAccumulator(cols)
    for chunk in x.split(500):  # in pieces, as calibration batches arrive
        acc.add(chunk)
    return w, acc.H, x


def naive_obs(w, H, spec, damp=0.01):
    """The definition: for each column in turn, invert H over the columns not yet quantized and apply
    δ_F = −e / [H_F⁻¹]_qq · [H_F⁻¹]_q,F. No Cholesky, no batching."""
    W, cols = w.clone().double(), w.shape[1]
    H = H.double() + damp * H.diagonal().mean() * torch.eye(cols, dtype=torch.float64)
    scale, zero = scale_and_zero(grouped(w.float(), spec), spec.bits, spec.symmetric)
    Q = torch.zeros_like(W)
    for q in range(cols):
        hinv = torch.linalg.inv(H[q:, q:])  # [F, F], F = columns q … end
        Q[:, q] = round_to_grid(W[:, q : q + 1], scale[:, 0], zero[:, 0], spec.bits, spec.symmetric)[:, 0]
        e = W[:, q] - Q[:, q]
        W[:, q:] -= (e / hinv[0, 0])[:, None] * hinv[0][None, :]
    return Q.float()


def test_the_accumulator_is_twice_the_mean_outer_product():
    _, H, x = correlated_problem()
    assert torch.allclose(H, 2 * x.T @ x / len(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_cholesky_and_lazy_batches_equal_the_naive_obs_loop(block_size):
    w, H, _ = correlated_problem()
    spec = IntSpec(4, "channel")
    ours = gptq_quantize(w, H, GPTQConfig(spec, block_size=block_size))
    assert torch.allclose(ours, naive_obs(w, H, spec), atol=1e-4)


def test_uncorrelated_inputs_leave_nothing_to_compensate():
    w, _, _ = correlated_problem()
    spec = IntSpec(4, "group", 16)
    ours = gptq_quantize(w, 3 * torch.eye(64), GPTQConfig(spec, block_size=32))
    assert torch.equal(ours, fake_quantize(w, spec))  # H diagonal: GPTQ is exactly RTN


@pytest.mark.parametrize(
    "spec", [IntSpec(4, "channel"), IntSpec(3, "group", 16), IntSpec(4, "group", 16, False)]
)
def test_gptq_beats_rtn_on_the_layer_output(spec):
    for seed in range(3):
        w, H, _ = correlated_problem(seed=seed)
        gptq = layer_loss(w, gptq_quantize(w, H, GPTQConfig(spec, block_size=32)), H)
        rtn = layer_loss(w, fake_quantize(w, spec), H)
        assert gptq < 0.8 * rtn


def test_act_order_depends_on_the_inputs_not_on_how_columns_are_listed():
    w, H, _ = correlated_problem()
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(5))
    cfg = GPTQConfig(IntSpec(4, "channel"), block_size=16, act_order=True)
    shuffled = gptq_quantize(w[:, perm], H[perm][:, perm], cfg)
    assert torch.allclose(shuffled, gptq_quantize(w, H, cfg)[:, perm], atol=1e-5)


def test_static_groups_keep_each_original_group_on_one_grid():
    w, H, _ = correlated_problem()
    spec = IntSpec(3, "group", 16)
    q = gptq_quantize(w, H, GPTQConfig(spec, block_size=16, act_order=True, static_groups=True))
    for group in q.reshape(32, 4, 16).reshape(-1, 16):
        assert len(set(group.tolist())) <= 2**3


def test_dead_inputs_are_zeroed_and_nothing_breaks():
    w, H, _ = correlated_problem()
    H[:, 3] = 0
    H[3, :] = 0
    q = gptq_quantize(w, H, GPTQConfig(IntSpec(4, "group", 16), block_size=32))
    assert torch.isfinite(q).all() and (q[:, 3] == 0).all()


def test_record_traces_every_column_with_its_rounding_error():
    w, H, _ = correlated_problem(rows=4, cols=8)
    record = []
    q = gptq_quantize(w, H, GPTQConfig(IntSpec(4, "channel"), block_size=8), record=record)
    assert [r["column"] for r in record] == list(range(8))
    assert torch.allclose(record[-1]["weights"], q)  # after the last column, everything is quantized
    assert all(r["error"].abs().max() > 0 for r in record)
