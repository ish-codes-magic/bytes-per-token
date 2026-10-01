"""A reference KV-cache policy for nanoserve: quantized storage (simulated) and StreamingLLM eviction.

Nothing here makes nanoserve faster. It changes what attention *sees* exactly as a real quantized or evicting
cache would, so quality (KL, needle-in-a-haystack) can be measured before any kernel exists:

- **Quantized storage.** Keys and values are rounded to the cache's grid as they're written, then read back
  dequantized. Attention runs on what the cache would actually hold.
  - Per-token (values, and naive keys): one scale per token's head vector (or per `group` elements of it).
  - Per-channel (KIVI's keys): one scale per channel, shared by `group` consecutive tokens. Keys have a few
    channels that are large for *every* token; per-channel scales isolate them, per-token scales can't.
    A group can only be quantized once all its tokens exist, so the last `T mod group` tokens of each write
    stay in full precision: KIVI's "residual". Prefill chunks should be multiples of `group`.
  - Rotated keys (QuaRot-style): keys are multiplied by a random Hadamard matrix R before rounding and by Rᵀ
    after. The rotation spreads each outlier channel over all channels, so a per-token grid fits better.
    Rotating back is exact, so only the rounding error changes.
  - FP8: E4M3 with a per-tensor scale of 1.0, which is vLLM 0.30's default for `--kv-cache-dtype fp8`.
- **Eviction (StreamingLLM).** Each query sees only the first `sinks` tokens (the attention sinks) and the
  last `window` ones. Original positions are kept for RoPE; within Qwen3's 32k training length that matters
  little, and it keeps the mask a pure function of positions.

Shapes: k, v [batch, kv_heads, tokens, head_dim]; mask [batch, 1, q_len, kv_len].
"""

from __future__ import annotations

import torch

from fastserve.kv.sizing import KVSpec, TensorQuant
from fastserve.quant.rotation import random_hadamard
from fastserve.quant.rtn import IntSpec, fake_quantize

FP8_MAX = 448.0  # largest finite E4M3 value


def _round_rows(rows: torch.Tensor, tq: TensorQuant) -> torch.Tensor:
    """Quantize-dequantize each row of a 2-D tensor with its own scale (and zero-point)."""
    return fake_quantize(rows, IntSpec(bits=tq.bits, granularity="channel", symmetric=tq.symmetric))


def store(x: torch.Tensor, tq: TensorQuant | None) -> torch.Tensor:
    """What a cache in format `tq` holds for x [B, H, T, D] (dequantized, same dtype). None = BF16 as is."""
    if tq is None:
        return x
    if tq.kind == "fp8":  # scale 1.0; the cache write saturates instead of overflowing
        return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(x.dtype)
    b, h, t, d = x.shape
    if tq.axis == "token":  # groups of `group` elements along each token's head vector
        return _round_rows(x.reshape(-1, tq.group), tq).reshape(b, h, t, d)
    full = t - t % tq.group  # channel axis: whole groups of tokens only; the remainder stays full precision
    if full == 0:
        return x
    blocks = x[:, :, :full].reshape(b, h, full // tq.group, tq.group, d).transpose(-1, -2)  # [B,H,n,D,group]
    rounded = _round_rows(blocks.reshape(-1, tq.group), tq).reshape(blocks.shape).transpose(-1, -2)
    return torch.cat([rounded.reshape(b, h, full, d), x[:, :, full:]], dim=2)


class KVPolicy:
    """The policy one Attention layer applies: `store` before the cache, `visible` on the mask."""

    def __init__(self, spec: KVSpec, head_dim: int, device: torch.device | str, seed: int = 0):
        self.spec = spec
        self.rotation = None
        if spec.rotate_keys:
            self.rotation = random_hadamard(head_dim, seed=seed, dtype=torch.float32).to(device)  # [D, D]

    def store(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotation is not None and self.spec.keys is not None:
            r = self.rotation
            k = (store((k.float() @ r).to(k.dtype), self.spec.keys).float() @ r.T).to(k.dtype)
        else:
            k = store(k, self.spec.keys)
        return k, store(v, self.spec.values)

    def visible(self, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Hide every slot an evicting cache would have dropped by the time each query runs."""
        if self.spec.window is None:
            return mask
        slots = torch.arange(mask.shape[-1], device=mask.device)  # [kv_len]
        recent = slots[None, None, :] > positions[:, :, None] - self.spec.window  # [B, Q, kv_len]
        sink = slots < (self.spec.sinks or 0)
        return mask & (recent | sink)[:, None]


def apply_kv_policy(model: torch.nn.Module, spec: KVSpec | None) -> torch.nn.Module:
    """Give every attention layer the policy (None removes it). Returns the model, changed in place."""
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.kv_policy = (
            None if spec is None else KVPolicy(spec, attn.cfg.head_dim, attn.q_proj.weight.device, i)
        )
    return model
