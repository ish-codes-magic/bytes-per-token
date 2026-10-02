"""A KV cache for nanoserve that really stores 4- or 8-bit codes, read on decode steps by kernel 2.

M5's `KVPolicy` only *simulated* a quantized cache: it rounded keys and values and stored the result in BF16,
which was enough to measure quality and saved no bytes. This cache holds the codes themselves
(`reference.QuantKV`), so its memory is what M5's sizing model promised, and decode attention reads them
without ever writing them back out as floats.

How a write and a read go (g = the key group, 32 tokens):

- **Write.** KIVI gives every channel of the keys one grid per g tokens, and a grid needs all g tokens. New
  tokens therefore wait in a small full-precision *tail* (fewer than g tokens) and are quantized, a whole
  group at a time, as soon as their group is complete. A prefill chunk that is a multiple of g is quantized
  at once, exactly as M5's simulation did.
- **Decode read (one new token).** Kernel 2 attends over the codes and returns one partial result per split;
  plain PyTorch attends over the tail; `merge_partials` joins them. The split-KV merge is what lets two
  storage formats share one attention.
- **Prefill read (many new tokens).** The unfused reference path: dequantize, then PyTorch's fused attention.
  The kernel is a decode kernel; prefill is compute-bound and not where the cache's bytes matter.

Limits, on purpose: rows advance in lockstep (every row is written at the same positions, so one token count
serves the whole batch), the cache is append-only (no rollback, so no speculative decoding), and the layout
is contiguous. A paged cache would need the block table inside the kernel, as vLLM's kernels have.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from fastserve.engine.attention import attention_sdpa, causal_mask
from fastserve.engine.config import ModelConfig
from fastserve.engine.kv_cache import KVCache
from fastserve.kernels import reference
from fastserve.kernels.reference import QuantKV

# (q [B, Hq, D], kv, lengths [B], scale, tokens) → (out [B, Hq, S, D], lse [B, Hq, S])
DecodeFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def reference_decode(
    q: torch.Tensor, kv: QuantKV, lengths: torch.Tensor, scale: float, *, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unfused decode read (dequantize, then attend): what the cache uses when no kernel is given."""
    k, v = reference.dequantize_kv(kv, tokens)
    out, lse = reference.partial_attention(q, k, v, lengths, scale)
    return out[:, :, None], lse[:, :, None]


@dataclass(frozen=True)
class QuantMeta:
    positions: torch.Tensor  # [batch, q_len]
    start: int  # tokens in the cache before this write
    q_len: int
    lengths: torch.Tensor  # [batch] int32: tokens held as codes once this write is in


class QuantizedKVCache(KVCache):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        max_batch: int,
        max_len: int,
        bits: int,
        dtype: torch.dtype,
        device: torch.device | str,
        group: int = 32,
        decode: DecodeFn = reference_decode,
    ):
        """bits 4 or 8, or 16 for the control: no quantization, but the same kernel and the same code path.

        `decode` is the decode-step attention over the codes: `kernels.kv_attention.decode_attention` on a
        GPU, the reference by default (so the cache itself is testable on a CPU).
        """
        if bits not in (4, 8, 16):
            raise ValueError(f"bits must be 4, 8 or 16, got {bits}")
        self.bits, self.group, self.decode, self.max_batch = bits, group, decode, max_batch
        heads, d = cfg.num_kv_heads, cfg.head_dim
        tokens = -(-max_len // group) * group  # whole groups
        self.max_len = tokens

        def layer() -> QuantKV:
            if bits == 16:
                shape = (max_batch, heads, tokens, d)
                return QuantKV(
                    16,
                    group,
                    k_codes=torch.zeros(shape, dtype=dtype, device=device),
                    v_codes=torch.zeros(shape, dtype=dtype, device=device),
                )
            codes = (max_batch, heads, tokens, d // 2 if bits == 4 else d)  # 4 bits: two channels per byte

            def grid(*shape: int) -> torch.Tensor:
                return torch.zeros(shape, dtype=torch.float16, device=device)

            return QuantKV(
                bits,
                group,
                k_codes=torch.zeros(codes, dtype=torch.uint8, device=device),
                v_codes=torch.zeros(codes, dtype=torch.uint8, device=device),
                k_scale=grid(max_batch, heads, tokens // group, d),
                k_zero=grid(max_batch, heads, tokens // group, d),
                v_scale=grid(max_batch, heads, tokens),
                v_zero=grid(max_batch, heads, tokens),
            )

        self.layers = [layer() for _ in range(cfg.num_layers)]
        tail = (max_batch, heads, group, d)  # the tokens whose key group is not complete yet
        self.tail_k = [torch.zeros(tail, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.tail_v = [torch.zeros(tail, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.length = 0  # tokens written so far, the same for every row
        self.device = device

    def bytes_per_token(self) -> float:
        """Bytes one token occupies across all layers and KV heads, as allocated (grids included)."""
        return sum(kv.bytes_per_token() for kv in self.layers)

    def prepare(self, rows: torch.Tensor, positions: torch.Tensor) -> QuantMeta:
        b, q_len = positions.shape
        start = self.length
        expected = torch.arange(start, start + q_len, device=positions.device)
        if b != self.max_batch or not bool((positions == expected).all()):
            raise ValueError(
                f"QuantizedKVCache is append-only and lockstep: expected all {self.max_batch} rows at "
                f"positions {start}..{start + q_len - 1}"
            )
        if start + q_len > self.max_len:
            raise ValueError(f"position {start + q_len - 1} exceeds the cache length {self.max_len}")
        self.length = start + q_len
        stored = self.length - self.length % self.group
        lengths = torch.full((b,), stored, dtype=torch.int32, device=positions.device)
        return QuantMeta(positions=positions, start=start, q_len=q_len, lengths=lengths)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, meta: QuantMeta):
        raise NotImplementedError("a quantized cache never returns float keys and values: use attend()")

    def _append(self, layer: int, k: torch.Tensor, v: torch.Tensor, meta: QuantMeta) -> tuple[int, int]:
        """Add new k, v [B, Hkv, q_len, D]; quantize every completed group. Returns (stored, waiting)."""
        g, kv = self.group, self.layers[layer]
        done, waiting = meta.start - meta.start % g, meta.start % g
        pending_k = torch.cat([self.tail_k[layer][:, :, :waiting], k], dim=2)  # [B, Hkv, waiting + q_len, D]
        pending_v = torch.cat([self.tail_v[layer][:, :, :waiting], v], dim=2)
        full = pending_k.shape[2] // g * g  # tokens that now form whole groups
        if full:
            new = reference.quantize_kv(pending_k[:, :, :full], pending_v[:, :, :full], self.bits, g)
            kv.k_codes[:, :, done : done + full] = new.k_codes
            kv.v_codes[:, :, done : done + full] = new.v_codes
            if self.bits != 16:
                kv.k_scale[:, :, done // g : (done + full) // g] = new.k_scale
                kv.k_zero[:, :, done // g : (done + full) // g] = new.k_zero
                kv.v_scale[:, :, done : done + full] = new.v_scale
                kv.v_zero[:, :, done : done + full] = new.v_zero
        rest = pending_k.shape[2] - full
        self.tail_k[layer][:, :, :rest] = pending_k[:, :, full:]
        self.tail_v[layer][:, :, :rest] = pending_v[:, :, full:]
        return done + full, rest

    def attend(
        self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, meta: QuantMeta, scale: float
    ) -> torch.Tensor:
        """Store this step's k, v, then attend: q [B, Hq, q_len, D] → [B, Hq, q_len, D].

        New tokens are stored *before* attention, so a query sees its own chunk as the cache holds it (M5's
        simulation did the same, which keeps the two comparable).
        """
        stored, rest = self._append(layer, k, v, meta)
        kv = self.layers[layer]
        tail_k, tail_v = self.tail_k[layer][:, :, :rest], self.tail_v[layer][:, :, :rest]

        if meta.q_len == 1:  # decode: the kernel over the codes, plain attention over the tail, merged
            query = q[:, :, 0]  # [B, Hq, D]
            outs, lses = [], []
            if stored:
                out, lse = self.decode(query, kv, meta.lengths, scale, tokens=stored)
                outs.append(out)
                lses.append(lse)
            if rest:
                everything = torch.full_like(meta.lengths, rest)
                out, lse = reference.partial_attention(query, tail_k, tail_v, everything, scale)
                outs.append(out[:, :, None])
                lses.append(lse[:, :, None])
            merged = reference.merge_partials(torch.cat(outs, dim=2), torch.cat(lses, dim=2))
            return merged[:, :, None].to(q.dtype)

        # prefill: the unfused path
        k_all, v_all = reference.dequantize_kv(kv, stored)
        k_all = torch.cat([k_all.to(q.dtype), tail_k], dim=2)  # [B, Hkv, start + q_len, D]
        v_all = torch.cat([v_all.to(q.dtype), tail_v], dim=2)
        mask = causal_mask(meta.positions, meta.start + meta.q_len)
        return attention_sdpa(q, k_all, v_all, mask, scale)
