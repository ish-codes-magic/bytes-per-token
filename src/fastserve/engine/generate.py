"""Generation loops.

- `generate`: static batching. Pad the prompts into one batch, prefill once, decode until *every* sequence is
  done. Sequences that finish early keep occupying their row, which is wasted work.
- `ContinuousBatcher`: iteration-level scheduling (Orca, vLLM). At every step, finished requests leave,
  waiting requests join, and all running requests share ONE batched decode step. It runs over a paged KV
  cache, so memory follows the requests that are actually running.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from fastserve.engine.kv_cache import ContiguousKVCache, PagedKVCache
from fastserve.engine.model import CausalLM
from fastserve.engine.sampler import SamplingParams, sample


def _is_done(output: list[int], params: SamplingParams) -> bool:
    return len(output) >= params.max_new_tokens or (bool(output) and output[-1] in params.stop_token_ids)


@torch.inference_mode()
def generate(
    model: CausalLM,
    prompts: list[list[int]],
    params: SamplingParams,
    *,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Static batching over a contiguous cache. Returns the generated token ids for each prompt."""
    weight = model.lm_head.weight
    b = len(prompts)
    lengths = torch.tensor([len(p) for p in prompts], device=weight.device)
    max_prompt = int(lengths.max())
    cache = ContiguousKVCache(
        model.config,
        max_batch=b,
        max_len=max_prompt + params.max_new_tokens,
        dtype=weight.dtype,
        device=weight.device,
    )

    # Prefill: right-pad to one [B, max_prompt] batch. Padding writes junk KV beyond each prompt's end, but
    # those slots sit past the sequence's position, so the causal mask hides them until they're overwritten.
    input_ids = torch.zeros(b, max_prompt, dtype=torch.long, device=weight.device)
    for i, prompt in enumerate(prompts):
        input_ids[i, : len(prompt)] = torch.tensor(prompt, device=weight.device)
    positions = torch.arange(max_prompt, device=weight.device).expand(b, -1)
    logits = model(input_ids, positions, cache, select=lengths - 1)  # each prompt's last token: [B, vocab]

    outputs: list[list[int]] = [[] for _ in prompts]
    next_positions = lengths.clone()
    zeros = torch.zeros(b, dtype=torch.long, device=weight.device)
    while True:
        tokens = sample(logits, params, generator)  # [B]
        for output, token in zip(outputs, tokens.tolist(), strict=True):
            if not _is_done(output, params):
                output.append(token)
        if all(_is_done(output, params) for output in outputs):
            return outputs
        logits = model(tokens[:, None], next_positions[:, None], cache, select=zeros)  # decode: 1 token each
        next_positions += 1


@torch.inference_mode()
def generate_long(
    model: CausalLM,
    prompt: list[int],
    params: SamplingParams,
    *,
    chunk: int = 512,
    generator: torch.Generator | None = None,
) -> list[int]:
    """One long prompt: prefill it `chunk` tokens at a time through a real cache, then decode.

    Chunking bounds the prefill's activation memory (a 32k-token prompt in one pass needs [32k × 32k]
    attention scores per head in the reference attention). The cache makes the result identical to one pass.
    """
    weight = model.lm_head.weight
    device = weight.device
    cache = ContiguousKVCache(
        model.config,
        max_batch=1,
        max_len=len(prompt) + params.max_new_tokens,
        dtype=weight.dtype,
        device=device,
    )
    ids = torch.tensor(prompt, device=device)[None]  # [1, P]
    for start in range(0, len(prompt), chunk):
        piece = ids[:, start : start + chunk]
        positions = torch.arange(start, start + piece.shape[1], device=device)[None]
        logits = model(piece, positions, cache, select=torch.tensor([piece.shape[1] - 1], device=device))
    output: list[int] = []
    zero = torch.zeros(1, dtype=torch.long, device=device)
    while True:
        token = sample(logits, params, generator)  # [1]
        output.append(int(token))
        if _is_done(output, params):
            return output
        position = torch.tensor([[len(prompt) + len(output) - 1]], device=device)
        logits = model(token[:, None], position, cache, select=zero)


@dataclass
class Request:
    id: Any
    prompt: list[int]
    params: SamplingParams
    arrival_step: int = 0  # lets tests and benchmarks simulate requests arriving over time
    output: list[int] = field(default_factory=list)


class ContinuousBatcher:
    """Admit → decode → retire, once per step, over a paged KV cache.

    Admission policy ("reserve a budget, allocate lazily"): a request is admitted only if the free blocks
    cover its worst case (prompt + max_new_tokens) *plus* what the running requests may still need. Blocks
    are then allocated only as sequences actually grow, so a running request never runs out mid-flight.
    (vLLM is more aggressive: it overcommits and preempts sequences when memory runs out.)
    """

    def __init__(
        self,
        model: CausalLM,
        cache: PagedKVCache,
        *,
        max_batch: int,
        generator: torch.Generator | None = None,
    ):
        self.model, self.cache, self.max_batch, self.generator = model, cache, max_batch, generator
        self.device = model.lm_head.weight.device
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: dict[Any, list[int]] = {}
        self.step_count = 0
        self.log: list[dict[str, Any]] = []  # one snapshot per step, for the block-table animation

    def submit(self, request: Request) -> None:
        self.waiting.append(request)

    def run(self) -> dict[Any, list[int]]:
        while self.waiting or self.running:
            self.step()
        return self.finished

    @torch.inference_mode()
    def step(self) -> None:
        self._admit()
        self._decode()
        self._retire()
        self.log.append(
            {
                "step": self.step_count,
                "running": [r.id for r in self.running],
                "waiting": [r.id for r in self.waiting],
                "block_tables": {seq: list(table) for seq, table in self.cache.tables.items()},
                "free_blocks": len(self.cache.free_blocks),
            }
        )
        self.step_count += 1

    # -- the three phases ------------------------------------------------------------------------------------

    def _worst_case_blocks(self, r: Request) -> int:
        return self.cache.blocks_needed(len(r.prompt) + r.params.max_new_tokens)

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_batch:
            request = self.waiting[0]
            if request.arrival_step > self.step_count:
                return
            promised = sum(
                self._worst_case_blocks(r) - len(self.cache.tables.get(r.id, [])) for r in self.running
            )
            available = len(self.cache.free_blocks) - promised
            if self._worst_case_blocks(request) > available:
                if not self.running and self._worst_case_blocks(request) > self.cache.num_blocks:
                    raise ValueError(f"request {request.id!r} can never fit in the KV cache")
                return  # wait for running requests to finish and free their blocks
            self.waiting.popleft()
            self._prefill(request)

    def _prefill(self, r: Request) -> None:
        n = len(r.prompt)
        self.cache.reserve(r.id, n)
        ids = torch.tensor([r.prompt], device=self.device)
        positions = torch.arange(n, device=self.device)[None]
        logits = self.model(
            ids, positions, self.cache, rows=[r.id], select=torch.tensor([n - 1], device=self.device)
        )
        r.output.append(int(sample(logits, r.params, self.generator)[0]))
        self.running.append(r)

    def _decode(self) -> None:
        active = [r for r in self.running if not _is_done(r.output, r.params)]
        if not active:
            return
        for r in active:  # each feeds its last token at position len(prompt) + len(output) − 1
            self.cache.reserve(r.id, len(r.prompt) + len(r.output))
        tokens = torch.tensor([[r.output[-1]] for r in active], device=self.device)
        positions = torch.tensor([[len(r.prompt) + len(r.output) - 1] for r in active], device=self.device)
        zeros = torch.zeros(len(active), dtype=torch.long, device=self.device)
        logits = self.model(tokens, positions, self.cache, rows=[r.id for r in active], select=zeros)
        for i, r in enumerate(active):  # per-request sampling, since requests may use different settings
            r.output.append(int(sample(logits[i : i + 1], r.params, self.generator)[0]))

    def _retire(self) -> None:
        for r in [r for r in self.running if _is_done(r.output, r.params)]:
            self.running.remove(r)
            self.cache.free(r.id)
            self.finished[r.id] = r.output
