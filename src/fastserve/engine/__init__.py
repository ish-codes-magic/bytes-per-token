"""nanoserve: a minimal, readable inference engine for Qwen3 (Llama-family) models in plain PyTorch.

It's the testbed for every technique in this project: small enough to read in one sitting, and checked
against Hugging Face `transformers` so we know it computes the same thing.

    config.py    model dimensions from config.json
    rope.py      rotary position embeddings
    attention.py causal masking + grouped-query attention
    kv_cache.py  contiguous and paged KV caches (same interface)
    model.py     the forward pass
    loader.py    weights from safetensors
    sampler.py   greedy / temperature / top-p
    generate.py  static batching and continuous batching
"""
