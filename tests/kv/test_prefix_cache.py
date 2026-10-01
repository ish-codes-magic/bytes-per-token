from fastserve.kv.prefix_cache import RadixCache, replay


def test_match_returns_the_longest_cached_prefix_and_its_payload():
    cache = RadixCache()
    assert cache.insert([1, 2, 3, 4], ["a", "b", "c", "d"]) == 4
    matched, pieces = cache.match([1, 2, 3, 9, 9])
    assert matched == 3 and sum(pieces, []) == ["a", "b", "c"]
    assert cache.match([7, 8]) == (0, [])


def test_divergent_sequences_split_an_edge_and_share_the_prefix():
    cache = RadixCache()
    cache.insert([1, 2, 3, 4], list("abcd"))
    assert cache.insert([1, 2, 5], list("abe")) == 1  # only the new token is stored
    assert cache.size == 5
    shared = cache.root.children[1]
    assert shared.tokens == (1, 2) and sorted(shared.children) == [3, 5]
    matched, pieces = cache.match([1, 2, 5, 6])
    assert matched == 3 and sum(pieces, []) == ["a", "b", "e"]
    assert cache.insert([1, 2, 3, 4], list("abcd")) == 0  # already cached


def test_lru_evicts_unused_leaves_and_never_a_shared_prefix_in_use():
    cache = RadixCache(capacity_tokens=6)
    cache.insert([1, 2, 3], list("xyz"))  # A
    cache.insert([1, 2, 4], list("xyw"))  # B shares [1, 2]
    cache.match([1, 2, 3])  # A is now more recent than B
    cache.insert([7, 8, 9], list("pqr"))  # 7 tokens > 6: evict the LRU leaf, B's [4]
    assert cache.size == 6 and cache.evicted_tokens == 1
    assert cache.match([1, 2, 4])[0] == 2
    assert cache.match([1, 2, 3])[0] == 3


def test_replay_counts_what_each_prompt_finds_cached():
    system = [100, 101, 102, 103]
    prompts = [system + [1, 2], system + [3], system + [1, 2, 9]]
    cache, cached = replay(prompts)
    assert cached == [0, 4, 6]
    assert cache.hit_tokens == 10 and cache.queried_tokens == 6 + 5 + 7
    root = cache.root.children[100]
    assert root.tokens == tuple(system) and root.hits == 2  # the two later prompts both passed through


def test_nanoserve_with_a_prefix_cache_generates_the_same_tokens(tiny_model):
    import pytest

    torch = pytest.importorskip("torch")
    from fastserve.engine.generate import generate
    from fastserve.engine.sampler import SamplingParams
    from fastserve.kv.prefix_generate import generate_with_prefix_cache

    g = torch.Generator().manual_seed(0)
    system = torch.randint(0, 256, (24,), generator=g).tolist()
    turns = [torch.randint(0, 256, (n,), generator=g).tolist() for n in (5, 9, 3)]
    prompts = [system + turns[0], system + turns[1], system + turns[0] + turns[2]]  # 3rd extends the 1st
    params = SamplingParams(max_new_tokens=8)
    plain = [generate(tiny_model, [p], params)[0] for p in prompts]
    cache = RadixCache()
    cached, reused = generate_with_prefix_cache(tiny_model, prompts, params, cache)
    assert cached == plain
    assert reused == [0, 24, 24 + 5]  # nothing, then the system prompt, then the whole first prompt
    assert cache.size == 24 + 5 + 9 + 3  # every distinct token run stored once
