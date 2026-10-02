"""Plain-PyTorch references for the Triton kernels: the storage formats, and the math written the slow way.

Everything a kernel computes is defined here first, in float32, one visible step at a time. The kernels are
tested against these functions, and nothing in this file needs a GPU.

**Kernel 1: RMSNorm + INT8 activation quantization.** `rms_norm_int8` is what vLLM's W8A8-INT8 path computes
in two ops (`rms_norm`, then `scaled_int8_quant`): normalize each token's hidden vector, then give it one
scale and round it to 8-bit codes.

**Kernel 2: decode attention over a quantized KV cache.** `QuantKV` is the cache one layer holds, in KIVI's
layout (M5 measured its quality by simulation; here the codes are real):

    keys     one grid per channel, shared by `group` consecutive tokens    codes + scale, zero per (group, d)
    values   one grid per token's head vector                              codes + scale, zero per token

    asymmetric grid   s = (max − min) / (2^b − 1)    z = round(−min / s)    code = clamp(round(x / s) + z)
    read back         x̂ = s · (code − z)

4-bit codes are packed two to a byte: channel d in the low nibble, channel d + D/2 in the high one. That
pairing (instead of neighbours d, d + 1) lets a kernel split one loaded byte into two half-vectors with a mask
and a shift, and never interleave them.

Decode attention for one query is softmax(q·kᵀ·scale)·v over the cached tokens. Split-KV cuts the tokens into
chunks, runs each chunk as its own attention (`partial_attention`: the chunk's output and the log of its
summed weights), and `merge_partials` recombines them. The merge is exact, not an approximation.

Shapes: B sequences, Hq query heads, Hkv KV heads, T cached tokens, D head dim, S splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fastserve.engine.attention import repeat_kv

INT8_MAX = 127.0


# ---- kernel 1: RMSNorm + INT8 quantization ---------------------------------------------------------------


def rms_norm_int8(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm, then per-token symmetric INT8: (codes int8 [..., d], scale float32 [..., 1]).

    y = x / sqrt(mean(x²) + eps) · weight        the model's RMSNorm, kept in float32 throughout
    scale = max|y| / 127                         one per token: the token's largest value maps to ±127
    codes = clamp(round(y / scale), −128, 127)   round half to even, as torch.round and vLLM do
    """
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    scale = (y.abs().amax(-1, keepdim=True) / INT8_MAX).clamp(min=1e-12)  # an all-zero token: no 0/0
    codes = torch.clamp(torch.round(y / scale), -128, 127).to(torch.int8)
    return codes, scale


def dequantize_int8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """What the next matmul sees: codes · scale, in float32."""
    return codes.float() * scale


# ---- kernel 2: the quantized KV cache --------------------------------------------------------------------


@dataclass
class QuantKV:
    """One layer's cached keys and values for B sequences, as stored.

    bits 4 or 8: uint8 codes plus float16 grids. bits 16: no quantization, the "codes" are the float16 or
    bfloat16 keys and values themselves and the grids are None (the control: same kernel, more bytes).

    k_codes, v_codes  [B, Hkv, T, D]       bits 8 and 16
                      [B, Hkv, T, D/2]     bits 4: channel d and d + D/2 share a byte
    k_scale, k_zero   [B, Hkv, T/group, D] float16: per channel, per group of tokens
    v_scale, v_zero   [B, Hkv, T]          float16: per token
    """

    bits: int
    group: int
    k_codes: torch.Tensor
    v_codes: torch.Tensor
    k_scale: torch.Tensor | None = None
    k_zero: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None
    v_zero: torch.Tensor | None = None

    @property
    def tokens(self) -> int:
        return self.k_codes.shape[2]

    @property
    def head_dim(self) -> int:
        return self.k_codes.shape[3] * (2 if self.bits == 4 else 1)

    def bytes_per_token(self) -> float:
        """Bytes one token occupies across this layer's KV heads, grids included."""
        tensors = [self.k_codes, self.v_codes, self.k_scale, self.k_zero, self.v_scale, self.v_zero]
        total = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        return total / (self.k_codes.shape[0] * self.tokens)


def _grid(x: torch.Tensor, bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The asymmetric grid for the values along `dim` of x (float32): (scale, zero), both as stored.

    The scale is rounded to float16 *before* the codes are computed, so codes and scale stay consistent:
    reading back with the stored scale reproduces exactly what was rounded.
    """
    lo = x.amin(dim, keepdim=True).clamp(max=0)  # the range always includes 0, so a real 0 stays 0
    hi = x.amax(dim, keepdim=True).clamp(min=0)
    scale = ((hi - lo) / (2**bits - 1)).to(torch.float16)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # all zeros: any step represents them
    zero = torch.round(-lo / scale.float()).clamp(0, 2**bits - 1).to(torch.float16)
    return scale, zero


def _codes(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale.float()) + zero.float(), 0, 2**bits - 1).to(torch.uint8)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """4-bit codes [..., D] → bytes [..., D/2]: channel d low, channel d + D/2 high."""
    half = codes.shape[-1] // 2
    return codes[..., :half] | (codes[..., half:] << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Bytes [..., D/2] → 4-bit codes [..., D]."""
    return torch.cat([packed & 0xF, packed >> 4], dim=-1)


def quantize_kv(k: torch.Tensor, v: torch.Tensor, bits: int, group: int = 32) -> QuantKV:
    """Store k, v [B, Hkv, T, D] in `bits` bits. T must be a whole number of groups (see M5: KIVI keeps the
    last T mod group tokens in full precision until their group is complete)."""
    if bits == 16:
        return QuantKV(16, group, k.contiguous(), v.contiguous())
    b, h, t, d = k.shape
    if t % group:
        raise ValueError(f"{t} tokens are not a whole number of groups of {group}")
    k32, v32 = k.float(), v.float()

    blocks = k32.reshape(b, h, t // group, group, d)  # [B, Hkv, G, group, D]
    k_scale, k_zero = _grid(blocks, bits, dim=3)  # [B, Hkv, G, 1, D]: per channel, over a group's tokens
    k_codes = _codes(blocks, k_scale, k_zero, bits).reshape(b, h, t, d)

    v_scale, v_zero = _grid(v32, bits, dim=3)  # [B, Hkv, T, 1]: per token, over its head vector
    v_codes = _codes(v32, v_scale, v_zero, bits)

    if bits == 4:
        k_codes, v_codes = pack_nibbles(k_codes), pack_nibbles(v_codes)
    return QuantKV(
        bits,
        group,
        k_codes.contiguous(),
        v_codes.contiguous(),
        k_scale.squeeze(3).contiguous(),
        k_zero.squeeze(3).contiguous(),
        v_scale.squeeze(3).contiguous(),
        v_zero.squeeze(3).contiguous(),
    )


def dequantize_kv(kv: QuantKV, tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The first `tokens` keys and values the cache holds, in float32: [B, Hkv, tokens, D] each.

    This is the unfused path: it reads the codes and writes a float tensor 4–8× larger, which attention
    then reads again. Kernel 2 exists to skip that round trip.
    """
    t = kv.tokens if tokens is None else tokens
    if kv.bits == 16:
        return kv.k_codes[:, :, :t].float(), kv.v_codes[:, :, :t].float()
    k_codes, v_codes = kv.k_codes[:, :, :t], kv.v_codes[:, :, :t]
    if kv.bits == 4:
        k_codes, v_codes = unpack_nibbles(k_codes), unpack_nibbles(v_codes)
    groups = -(-t // kv.group)  # each token reads its group's grid
    k_scale = kv.k_scale[:, :, :groups].float().repeat_interleave(kv.group, dim=2)[:, :, :t]
    k_zero = kv.k_zero[:, :, :groups].float().repeat_interleave(kv.group, dim=2)[:, :, :t]
    k = k_scale * (k_codes.float() - k_zero)
    v = kv.v_scale[:, :, :t, None].float() * (v_codes.float() - kv.v_zero[:, :, :t, None].float())
    return k, v


# ---- kernel 2: decode attention, whole and in parts ------------------------------------------------------


def partial_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention of one query per head over one stretch of tokens: (out [B, Hq, D], lse [B, Hq]).

    q [B, Hq, D]; k, v [B, Hkv, T, D]; lengths [B]: row b sees its first lengths[b] tokens.
    `out` is this stretch's own softmax-weighted average of v. `lse` is log Σ exp(score): how much weight
    the stretch would get next to other stretches. A row that sees no token returns out 0 and lse −inf.
    """
    group = q.shape[1] // k.shape[1]
    k, v = repeat_kv(k.float(), group), repeat_kv(v.float(), group)  # [B, Hq, T, D]
    scores = torch.einsum("bhd,bhtd->bht", q.float(), k) * scale  # [B, Hq, T]
    visible = torch.arange(k.shape[2], device=q.device)[None, :] < lengths[:, None]  # [B, T]
    scores = scores.masked_fill(~visible[:, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B, Hq]
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)  # softmax of all −inf is NaN: no tokens, no output
    return torch.einsum("bht,bhtd->bhd", probs, v), lse


def merge_partials(outs: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """Combine S partial attentions into the attention over all their tokens: [B, Hq, S, D] → [B, Hq, D].

    Each part's share of the total weight is exp(lse_s) / Σ exp(lse): a softmax over the parts.
    """
    share = torch.softmax(lses, dim=-1).nan_to_num(0.0)  # [B, Hq, S]
    return (share[..., None] * outs).sum(dim=2)


def decode_attention(
    q: torch.Tensor, kv: QuantKV, lengths: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode attention over a quantized cache, the unfused way: dequantize everything, then attend.

    Returns (out [B, Hq, 1, D], lse [B, Hq, 1]): a single part, the same shape kernel 2 returns with one
    split, so both go through `merge_partials`.
    """
    k, v = dequantize_kv(kv)
    out, lse = partial_attention(q, k, v, lengths, scale)
    return out[:, :, None], lse[:, :, None]
