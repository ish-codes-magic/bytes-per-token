"""Batching must not change the answer: batched and continuous-batched greedy == each prompt on its own."""

import pytest

torch = pytest.importorskip("torch")

from fastserve.engine.generate import ContinuousBatcher, Request, generate  # noqa: E402
from fastserve.engine.kv_cache import PagedKVCache  # noqa: E402
from fastserve.engine.sampler import SamplingParams  # noqa: E402

PROMPTS = [[5, 17, 42, 9, 101], [7, 7, 3, 250, 11, 64, 8, 90, 1], [33, 2, 199]]
GREEDY = SamplingParams(max_new_tokens=6)


def _alone(model, prompt, params=GREEDY):
    return generate(model, [prompt], params)[0]


def test_static_batch_equals_one_prompt_at_a_time(tiny_model):
    assert generate(tiny_model, PROMPTS, GREEDY) == [_alone(tiny_model, p) for p in PROMPTS]


def test_generation_stops_at_a_stop_token(tiny_model):
    first = _alone(tiny_model, PROMPTS[0])[0]
    params = SamplingParams(max_new_tokens=6, stop_token_ids=(first,))
    assert generate(tiny_model, [PROMPTS[0]], params) == [[first]]


def test_continuous_batching_equals_independent_runs(tiny_model):
    prompts = [*PROMPTS, [9, 8, 7, 6], [120, 121]]
    # A small pool and max_batch=2 force queueing, lazy block allocation, and reuse of freed blocks.
    cache = PagedKVCache(tiny_model.config, num_blocks=10, block_size=4, dtype=torch.float32, device="cpu")
    batcher = ContinuousBatcher(tiny_model, cache, max_batch=2)
    for i, (prompt, arrival) in enumerate(zip(prompts, [0, 0, 1, 3, 4], strict=True)):
        batcher.submit(Request(id=i, prompt=prompt, params=GREEDY, arrival_step=arrival))
    finished = batcher.run()

    assert finished == {i: _alone(tiny_model, p) for i, p in enumerate(prompts)}
    assert len(cache.free_blocks) == cache.num_blocks  # every block came back
    assert max(len(s["running"]) for s in batcher.log) == 2
    assert any(s["waiting"] for s in batcher.log)  # some requests really had to wait


def test_continuous_batcher_rejects_a_request_that_can_never_fit(tiny_model):
    cache = PagedKVCache(tiny_model.config, num_blocks=2, block_size=2, dtype=torch.float32, device="cpu")
    batcher = ContinuousBatcher(tiny_model, cache, max_batch=1)
    batcher.submit(Request(id=0, prompt=[1, 2, 3], params=GREEDY))  # needs 5 blocks worst case
    with pytest.raises(ValueError, match="never fit"):
        batcher.run()
