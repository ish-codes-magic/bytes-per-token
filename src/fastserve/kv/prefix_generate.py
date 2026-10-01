"""nanoserve generation with a prefix cache: reuse cached keys/values, prefill only the uncached suffix.

One prompt at a time (the point is correctness, not throughput):
1. Look up the prompt in the radix tree. The last prompt token is always recomputed, because its logits
   start decoding, so at most `len(prompt) − 1` tokens are reused.
2. Copy the matched keys/values into a fresh cache at positions [0, matched).
3. Prefill tokens [matched, len) at their real positions, then decode as usual.
4. Insert the prompt's keys/values into the tree for later requests.

Because attention is causal, a token's K and V depend only on the tokens before it, so reused prefixes give
the same outputs as recomputing them (up to floating-point summation order).
"""

from __future__ import annotations

import torch

from fastserve.engine.generate import _is_done
from fastserve.engine.kv_cache import ContiguousKVCache
from fastserve.engine.model import CausalLM
from fastserve.engine.sampler import SamplingParams, sample
from fastserve.kv.prefix_cache import RadixCache


class KVSpan:
    """Keys and values of a run of tokens, every layer: k[layer], v[layer] are [kv_heads, tokens, head_dim].

    Slicing selects tokens, which is what the radix tree needs to split an edge.
    """

    def __init__(self, k: list[torch.Tensor], v: list[torch.Tensor]):
        self.k, self.v = k, v

    def __getitem__(self, s: slice) -> KVSpan:
        return KVSpan([x[:, s] for x in self.k], [x[:, s] for x in self.v])

    def __len__(self) -> int:
        return self.k[0].shape[1]


@torch.inference_mode()
def generate_with_prefix_cache(
    model: CausalLM,
    prompts: list[list[int]],
    params: SamplingParams,
    prefix_cache: RadixCache,
    *,
    generator: torch.Generator | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Generate for each prompt in turn. Returns (outputs, tokens reused from the cache per prompt)."""
    weight = model.lm_head.weight
    cfg, device = model.config, weight.device
    outputs, reused = [], []
    for prompt in prompts:
        matched, pieces = prefix_cache.match(prompt[:-1])
        cache = ContiguousKVCache(
            cfg, max_batch=1, max_len=len(prompt) + params.max_new_tokens, dtype=weight.dtype, device=device
        )
        if matched:
            for layer in range(cfg.num_layers):  # [kv_heads, matched, head_dim] into slot 0
                cache.k[layer][0, :, :matched] = torch.cat([p.k[layer] for p in pieces], dim=1)
                cache.v[layer][0, :, :matched] = torch.cat([p.v[layer] for p in pieces], dim=1)
        suffix = torch.tensor(prompt[matched:], device=device)[None]  # [1, len − matched]
        positions = torch.arange(matched, len(prompt), device=device)[None]
        logits = model(suffix, positions, cache, select=torch.tensor([suffix.shape[1] - 1], device=device))

        output: list[int] = []
        position = len(prompt)
        while True:
            token = sample(logits, params, generator)  # [1]
            output.append(int(token))
            if _is_done(output, params):
                break
            logits = model(
                token[:, None],
                torch.tensor([[position]], device=device),
                cache,
                select=torch.zeros(1, dtype=torch.long, device=device),
            )
            position += 1
        outputs.append(output)
        reused.append(matched)
        n = len(prompt)
        span = KVSpan(
            [cache.k[i][0, :, :n].clone() for i in range(cfg.num_layers)],
            [cache.v[i][0, :, :n].clone() for i in range(cfg.num_layers)],
        )
        prefix_cache.insert(prompt, span)
    return outputs, reused
