"""Kernel 2: decode attention that reads a quantized KV cache directly.

**What it computes.** For one new token per sequence, softmax(q·kᵀ·scale)·v over the tokens in the cache,
where the cache holds 4- or 8-bit codes (`reference.QuantKV`, KIVI's layout: keys per channel in groups of
32 tokens, values per token). Same result as `reference.decode_attention`.

**What it fuses.** Dequantization into the attention itself. The unfused path reads the codes, writes a
float tensor 4–8× larger, and attention reads that again. Here the codes are the only thing read, and
nothing the size of the cache is ever written. The trick is to move each grid to the small side of its
product, so no key or value is ever dequantized element by element:

    score_t = Σ_d q_d · s_d · (code_td − z_d) = Σ_d (q_d · s_d) · code_td − Σ_d q_d · s_d · z_d
              the key grid (s, z) is per channel and shared by a group of tokens, so it folds into the
              query once per group: one scaled query and one bias, then a plain dot product with the codes

    out     = Σ_t p_t · s_t · (code_t − z_t)  = Σ_t (p_t · s_t) · code_t − Σ_t p_t · s_t · z_t
              the value grid is per token, so it folds into the attention weight p_t

**Tiling.** One program per (sequence, KV head, split). Qwen3's two query heads per KV head are handled by
the *same* program, so each cached byte is read once, not once per query head. Inside a program the split's
tokens stream through in blocks of one key group (32 tokens × D channels), with a running maximum, weight
sum and output: the online softmax of FlashAttention. Each split writes its own partial result and
`reference.merge_partials` combines them exactly.

Splitting is what gives a single long sequence parallelism: without it, batch 1 would be 8 programs (one per
KV head) on a GPU with 58 SMs.

**Bytes moved per token and KV head** (D = 128):

    BF16 cache                       K 256 + V 256                             = 512
    INT8, dequantize then attend     read 276, write 512, read 512             = 1300
    INT8, this kernel                K 128 + V 128 + grids 20                  = 276
    INT4, this kernel                K 64 + V 64 + grids 20                    = 148     (3.5× under BF16)

**Expected bound.** About 1,000 FLOPs per token and KV head against 148–512 bytes: 2–7 FLOPs per byte, far
left of the ridge, so memory-bound at long context. At short context it is launch-bound: two launches here
(kernel, merge) against PyTorch's handful.
"""

from __future__ import annotations

import functools

import torch

from fastserve.kernels.reference import QuantKV, merge_partials


@functools.cache
def _kernel():
    """Built lazily so importing this module never requires Triton or a GPU."""
    import triton
    import triton.language as tl

    @triton.jit
    def decode_attention_kernel(
        q_ptr,  # [B, Hkv, REP, D] float32: the REP query heads served by each KV head
        k_codes_ptr,  # [B, Hkv, T, D] (BITS 8, 16) or [B, Hkv, T, D/2] (BITS 4)
        k_scale_ptr,  # [B, Hkv, T/GROUP, D] float16
        k_zero_ptr,
        v_codes_ptr,
        v_scale_ptr,  # [B, Hkv, T] float16
        v_zero_ptr,
        lengths_ptr,  # [B] int32: tokens visible to each sequence's query
        out_ptr,  # [B, Hkv, REP, S, D] float32, written: each split's own attention output
        lse_ptr,  # [B, Hkv, REP, S] float32, written: log Σ exp(score) over the split
        n_kv_heads,
        t_max,  # tokens the code tensors are laid out for (their size along T)
        n_groups,  # key groups the grid tensors are laid out for
        n_splits,
        sm_scale,
        SPLIT: tl.constexpr,  # tokens per program
        GROUP: tl.constexpr,  # tokens sharing a key grid; also the tokens processed per loop step
        HALF: tl.constexpr,  # D / 2
        REP: tl.constexpr,  # query heads per KV head
        BITS: tl.constexpr,  # 4, 8, or 16 (no quantization)
    ):
        bh = tl.program_id(0).to(tl.int64)  # (sequence, KV head); int64 because offsets can pass 2^31
        split = tl.program_id(1)
        n = tl.load(lengths_ptr + bh // n_kv_heads)

        d = tl.arange(0, HALF)  # channels of one half-vector
        r = tl.arange(0, REP)
        D = 2 * HALF
        ELEMS = HALF if BITS == 4 else D  # stored elements per token: a byte holds two 4-bit channels

        # Queries stay in registers for the whole program: [REP, HALF] for each half of the head vector.
        q_row = q_ptr + (bh * REP + r[:, None]) * D
        q_lo = tl.load(q_row + d[None, :]) * sm_scale
        q_hi = tl.load(q_row + HALF + d[None, :]) * sm_scale

        # Online softmax state, per query head: running max, Σ weights, Σ weights · values.
        m = tl.zeros([REP], dtype=tl.float32) - float("inf")
        total = tl.zeros([REP], dtype=tl.float32)
        acc_lo = tl.zeros([REP, HALF], dtype=tl.float32)
        acc_hi = tl.zeros([REP, HALF], dtype=tl.float32)

        first = split * SPLIT
        for start in range(first, tl.minimum(first + SPLIT, n), GROUP):
            t = start + tl.arange(0, GROUP)
            valid = t < n
            row = bh * t_max * ELEMS + t[:, None] * ELEMS  # [GROUP, 1] offset of each token's codes

            # ---- keys: codes [GROUP, HALF] per half, straight from memory -------------------------------
            if BITS == 4:
                packed = tl.load(k_codes_ptr + row + d[None, :], mask=valid[:, None], other=0)
                k_lo = (packed & 0xF).to(tl.float32)  # channel d
                k_hi = (packed >> 4).to(tl.float32)  # channel d + HALF
            else:
                k_lo = tl.load(k_codes_ptr + row + d[None, :], mask=valid[:, None], other=0).to(tl.float32)
                k_hi = tl.load(k_codes_ptr + row + HALF + d[None, :], mask=valid[:, None], other=0).to(
                    tl.float32
                )

            if BITS == 16:
                qs_lo, qs_hi = q_lo, q_hi
                bias = tl.zeros([REP], dtype=tl.float32)
            else:
                # This group's key grid folds into the query: [REP, HALF] scaled queries and one bias each.
                grid = (bh * n_groups + start // GROUP) * D
                qs_lo = q_lo * tl.load(k_scale_ptr + grid + d).to(tl.float32)[None, :]
                qs_hi = q_hi * tl.load(k_scale_ptr + grid + HALF + d).to(tl.float32)[None, :]
                z_lo = tl.load(k_zero_ptr + grid + d).to(tl.float32)
                z_hi = tl.load(k_zero_ptr + grid + HALF + d).to(tl.float32)
                bias = -(tl.sum(qs_lo * z_lo[None, :], axis=1) + tl.sum(qs_hi * z_hi[None, :], axis=1))

            # scores [REP, GROUP]: a dot product of each scaled query with each token's codes
            scores = tl.sum(qs_lo[:, None, :] * k_lo[None, :, :], axis=2)
            scores += tl.sum(qs_hi[:, None, :] * k_hi[None, :, :], axis=2)
            scores = tl.where(valid[None, :], scores + bias[:, None], float("-inf"))

            # ---- online softmax: rescale what was accumulated under the old maximum ---------------------
            m_new = tl.maximum(m, tl.max(scores, axis=1))
            keep = tl.exp(m - m_new)  # [REP]; 0 on the first block, where m is −inf
            p = tl.exp(scores - m_new[:, None])  # [REP, GROUP]

            # ---- values: weights [REP, GROUP] times codes [GROUP, HALF] ---------------------------------
            if BITS == 4:
                packed = tl.load(v_codes_ptr + row + d[None, :], mask=valid[:, None], other=0)
                v_lo = (packed & 0xF).to(tl.float32)
                v_hi = (packed >> 4).to(tl.float32)
            else:
                v_lo = tl.load(v_codes_ptr + row + d[None, :], mask=valid[:, None], other=0).to(tl.float32)
                v_hi = tl.load(v_codes_ptr + row + HALF + d[None, :], mask=valid[:, None], other=0).to(
                    tl.float32
                )

            if BITS == 16:
                w = p
                shift = tl.zeros([REP], dtype=tl.float32)
            else:
                # Each token's value grid folds into its attention weight.
                v_scale = tl.load(v_scale_ptr + bh * t_max + t, mask=valid, other=0.0).to(tl.float32)
                v_zero = tl.load(v_zero_ptr + bh * t_max + t, mask=valid, other=0.0).to(tl.float32)
                w = p * v_scale[None, :]
                shift = tl.sum(w * v_zero[None, :], axis=1)  # [REP]: the same for every channel

            acc_lo = (
                acc_lo * keep[:, None] + tl.sum(w[:, :, None] * v_lo[None, :, :], axis=1) - shift[:, None]
            )
            acc_hi = (
                acc_hi * keep[:, None] + tl.sum(w[:, :, None] * v_hi[None, :, :], axis=1) - shift[:, None]
            )
            total = total * keep + tl.sum(p, axis=1)
            m = m_new

        # This split's own normalized output, and how much weight it carries next to the other splits.
        # A split past the sequence's end ran no block: output 0, lse −inf, so the merge ignores it.
        norm = tl.where(total > 0, total, 1.0)
        out_row = out_ptr + ((bh * REP + r[:, None]) * n_splits + split) * D
        tl.store(out_row + d[None, :], acc_lo / norm[:, None])
        tl.store(out_row + HALF + d[None, :], acc_hi / norm[:, None])
        tl.store(lse_ptr + (bh * REP + r) * n_splits + split, m + tl.log(total))

    return decode_attention_kernel


def default_split(batch: int, kv_heads: int, tokens: int, group: int = 32, programs: int = 512) -> int:
    """Tokens per program: enough splits for about `programs` programs in total, never below 8 key groups.

    One split per (sequence, KV head) leaves a small batch with fewer programs than the GPU has SMs; too
    many splits and each program's fixed costs (loading the query, writing a partial) outweigh its work.
    """
    per_program = max(tokens * batch * kv_heads // programs, 8 * group)
    return max(group, 1 << (per_program.bit_length() - 1))  # round down to a power of two (a group multiple)


def decode_attention(
    q: torch.Tensor,
    kv: QuantKV,
    lengths: torch.Tensor,
    scale: float,
    *,
    tokens: int | None = None,
    split: int | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial attentions of one query per head over a quantized cache, one per split.

    q [B, Hq, D] (any float dtype); lengths [B] int32. `tokens` is the longest sequence's length when the
    caller knows it (the cache does): the kernel then launches splits for that many tokens only, without
    asking the GPU for `lengths.max()`. Returns (out [B, Hq, S, D], lse [B, Hq, S]) in float32;
    `merge_partials` turns them into the attention output, possibly together with partials from elsewhere
    (the cache's full-precision tail).
    """
    b, hq, d = q.shape
    kv_heads, t_max = kv.k_codes.shape[1], kv.tokens
    rep = hq // kv_heads
    tokens = t_max if tokens is None else tokens
    if split is None:
        split = default_split(b, kv_heads, tokens, kv.group)
    if split % kv.group:
        raise ValueError(f"split {split} must be a multiple of the key group {kv.group}")
    n_splits = max(-(-tokens // split), 1)
    out = torch.empty((b, hq, n_splits, d), dtype=torch.float32, device=q.device)
    lse = torch.empty((b, hq, n_splits), dtype=torch.float32, device=q.device)
    unused = kv.k_codes  # BITS 16 never reads the grids, but every pointer argument needs a tensor
    _kernel()[(b * kv_heads, n_splits)](
        q.float().contiguous(),
        kv.k_codes,
        kv.k_scale if kv.k_scale is not None else unused,
        kv.k_zero if kv.k_zero is not None else unused,
        kv.v_codes,
        kv.v_scale if kv.v_scale is not None else unused,
        kv.v_zero if kv.v_zero is not None else unused,
        lengths,
        out,
        lse,
        kv_heads,
        t_max,
        kv.k_scale.shape[2] if kv.k_scale is not None else 1,
        n_splits,
        scale,
        SPLIT=split,
        GROUP=kv.group,
        HALF=d // 2,
        REP=rep,
        BITS=kv.bits,
        num_warps=num_warps,
    )
    return out, lse


def attend(
    q: torch.Tensor, kv: QuantKV, lengths: torch.Tensor, scale: float, **launch: int | None
) -> torch.Tensor:
    """Decode attention over a quantized cache: [B, Hq, D]. The kernel, then the merge of its splits."""
    return merge_partials(*decode_attention(q, kv, lengths, scale, **launch))
