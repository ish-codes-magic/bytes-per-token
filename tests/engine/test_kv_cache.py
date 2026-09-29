"""The KV cache must not change the answer: prefill + one-token decode steps == one full forward pass."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.kv_cache import ContiguousKVCache, OutOfBlocksError, PagedKVCache  # noqa: E402

SEQ, PREFIX = 12, 5


def _make_cache(kind, cfg):
    if kind == "contiguous":
        return ContiguousKVCache(cfg, max_batch=2, max_len=SEQ, dtype=torch.float32, device="cpu"), None
    # Block size 4: each 12-token sequence spans 3 blocks, so decoding crosses block boundaries.
    cache = PagedKVCache(cfg, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu")
    for seq in ("a", "b"):
        cache.reserve(seq, SEQ)
    return cache, ["a", "b"]


@pytest.mark.parametrize("kind", ["contiguous", "paged"])
def test_prefill_then_decode_matches_full_forward(tiny_model, kind):
    ids = torch.randint(0, 256, (2, SEQ), generator=torch.Generator().manual_seed(3))
    cache, rows = _make_cache(kind, tiny_model.config)
    with torch.no_grad():
        full = tiny_model(ids)  # [2, SEQ, vocab], no cache
        positions = torch.arange(PREFIX).expand(2, -1)
        prefill = tiny_model(ids[:, :PREFIX], positions, cache, rows)
        torch.testing.assert_close(prefill, full[:, :PREFIX])
        for t in range(PREFIX, SEQ):  # one token at a time, crossing block boundaries for the paged cache
            step = tiny_model(ids[:, t : t + 1], torch.full((2, 1), t), cache, rows)
            torch.testing.assert_close(step[:, 0], full[:, t], atol=1e-5, rtol=1e-5)


def test_paged_allocator_hands_out_and_reuses_blocks(tiny_model):
    cache = PagedKVCache(tiny_model.config, num_blocks=4, block_size=2, dtype=torch.float32, device="cpu")
    cache.reserve("a", 3)  # 3 tokens -> 2 blocks
    cache.reserve("b", 4)  # 4 tokens -> 2 blocks
    assert len(cache.free_blocks) == 0
    assert not cache.can_reserve("c", 1)
    with pytest.raises(OutOfBlocksError):
        cache.reserve("c", 1)
    freed = set(cache.tables["a"])
    cache.free("a")
    cache.reserve("c", 4)
    assert set(cache.tables["c"]) == freed  # finished sequences' blocks are reused


def test_paged_cache_refuses_unreserved_positions(tiny_model):
    cache = PagedKVCache(tiny_model.config, num_blocks=4, block_size=2, dtype=torch.float32, device="cpu")
    cache.reserve("a", 2)
    with pytest.raises(ValueError, match="reserve"):
        cache.prepare(["a"], torch.tensor([[2]]))  # position 2 needs a second block
