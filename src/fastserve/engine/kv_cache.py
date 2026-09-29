"""KV caches: where each layer's keys and values live between decode steps.

Two implementations behind one interface, so the model doesn't care which it gets:

- ContiguousKVCache: one fixed [max_len] buffer per batch slot. Simple, but every slot reserves memory for the
  longest possible sequence.
- PagedKVCache: a shared pool of fixed-size blocks plus a *block table* per sequence (vLLM's PagedAttention
  idea, like virtual memory for the KV cache). Sequences grab blocks only as they grow and return them when
  done, so memory isn't reserved up front and finished sequences free space for new ones.

Per forward pass the model calls `prepare(...)` once (work shared by all layers: where to write, what to read,
in the spirit of vLLM's attention metadata), then `update(layer, ...)` once per layer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from fastserve.engine.config import ModelConfig


class KVCache(ABC):
    @abstractmethod
    def prepare(self, rows: Any, positions: torch.Tensor) -> Any:
        """Per-forward metadata. `rows` says which sequence each batch row belongs to."""

    @abstractmethod
    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, meta: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new k, v: [batch, kv_heads, q_len, head_dim] at their positions, and return every key/value
        written so far for these rows: [batch, kv_heads, kv_len, head_dim], where slot j holds position j."""


# ------------------------------------------------------------------------------------------------------------
# Contiguous
# ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContiguousMeta:
    rows: torch.Tensor  # [batch] batch-slot index of each row
    positions: torch.Tensor  # [batch, q_len]
    kv_len: int  # read slots [0, kv_len): the furthest position in this batch, + 1


class ContiguousKVCache(KVCache):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        max_batch: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        shape = (max_batch, cfg.num_kv_heads, max_len, cfg.head_dim)  # [slot, kv_heads, position, head_dim]
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.max_len = max_len

    def prepare(self, rows: torch.Tensor, positions: torch.Tensor) -> ContiguousMeta:
        kv_len = int(positions.max()) + 1
        if kv_len > self.max_len:
            raise ValueError(f"position {kv_len - 1} exceeds the cache length {self.max_len}")
        return ContiguousMeta(rows=rows, positions=positions, kv_len=kv_len)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, meta: ContiguousMeta):
        # Advanced indexing: [rows[:, None], :, positions] selects a [batch, q_len, kv_heads, head_dim] view.
        index = (meta.rows[:, None], slice(None), meta.positions)
        self.k[layer][index] = k.transpose(1, 2)
        self.v[layer][index] = v.transpose(1, 2)
        return self.k[layer][meta.rows, :, : meta.kv_len], self.v[layer][meta.rows, :, : meta.kv_len]


# ------------------------------------------------------------------------------------------------------------
# Paged
# ------------------------------------------------------------------------------------------------------------


class OutOfBlocksError(RuntimeError):
    """The pool has no free block: the scheduler must wait for sequences to finish (or preempt one)."""


@dataclass(frozen=True)
class PagedMeta:
    write_blocks: torch.Tensor  # [batch, q_len] physical block receiving each new token
    write_offsets: torch.Tensor  # [batch, q_len] slot inside that block
    block_table: torch.Tensor  # [batch, n_blocks] physical blocks holding logical blocks 0 … n_blocks − 1
    kv_len: int


class PagedKVCache(KVCache):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        shape = (num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)  # [block, slot, kv_heads, head_dim]
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks - 1, -1, -1))  # a stack: pop() hands out block 0 first
        self.tables: dict[Any, list[int]] = {}  # sequence id -> its block table
        self.device = device

    # -- memory management (host side) -----------------------------------------------------------------------

    def blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def can_reserve(self, seq_id: Any, num_tokens: int) -> bool:
        have = len(self.tables.get(seq_id, []))
        return self.blocks_needed(num_tokens) - have <= len(self.free_blocks)

    def reserve(self, seq_id: Any, num_tokens: int) -> None:
        """Make sure `seq_id` owns enough blocks to hold positions [0, num_tokens)."""
        table = self.tables.setdefault(seq_id, [])
        missing = self.blocks_needed(num_tokens) - len(table)
        if missing > len(self.free_blocks):
            raise OutOfBlocksError(f"need {missing} blocks, {len(self.free_blocks)} free")
        table.extend(self.free_blocks.pop() for _ in range(missing))

    def free(self, seq_id: Any) -> None:
        """Return a finished sequence's blocks to the pool."""
        self.free_blocks.extend(reversed(self.tables.pop(seq_id, [])))

    # -- the KVCache interface -------------------------------------------------------------------------------

    def prepare(self, rows: list[Any], positions: torch.Tensor) -> PagedMeta:
        kv_len = int(positions.max()) + 1
        n_blocks = self.blocks_needed(kv_len)
        tables = []
        for seq_id in rows:
            table = self.tables[seq_id]
            if len(table) * self.block_size < int(positions[len(tables)].max()) + 1:
                raise ValueError(f"sequence {seq_id!r} has too few blocks: call reserve() first")
            tables.append(table[:n_blocks] + [0] * (n_blocks - len(table)))  # pad; padded slots are masked
        block_table = torch.tensor(tables, dtype=torch.long, device=self.device)  # [batch, n_blocks]
        logical = positions // self.block_size
        return PagedMeta(
            write_blocks=torch.gather(block_table, 1, logical),
            write_offsets=positions % self.block_size,
            block_table=block_table,
            kv_len=kv_len,
        )

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, meta: PagedMeta):
        # Write: token (b, i) goes to slot write_offsets[b, i] of block write_blocks[b, i].
        self.k[layer][meta.write_blocks, meta.write_offsets] = k.transpose(1, 2)
        self.v[layer][meta.write_blocks, meta.write_offsets] = v.transpose(1, 2)
        # Read (reference version): gather each sequence's blocks back into a contiguous [kv_len] view.
        return self._gather(self.k[layer], meta), self._gather(self.v[layer], meta)

    def _gather(self, pool: torch.Tensor, meta: PagedMeta) -> torch.Tensor:
        b, n = meta.block_table.shape
        blocks = pool[meta.block_table]  # [batch, n_blocks, block_size, kv_heads, head_dim]
        flat = blocks.reshape(b, n * self.block_size, *pool.shape[2:])[:, : meta.kv_len]
        return flat.transpose(1, 2)  # [batch, kv_heads, kv_len, head_dim]
