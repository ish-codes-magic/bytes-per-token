"""KV-cache sizing from config.json: bytes per token in each storage format, and what fits in memory.

Stdlib only, so the performance model, the report and the laptop can all use it.

The cache holds, for every token, a key and a value vector per layer and KV head:

    bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes per element

A quantized cache also stores its scales (and zero-points), so its bytes per element are
`bits / 8 + (scale bits × (1 + has zero-point)) / (8 × elements per scale)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCALE_BITS = 16  # scales and zero-points are stored in 16-bit floats


@dataclass(frozen=True)
class TensorQuant:
    """How one of K or V is stored.

    kind:  "int" (a grid with scales), or "fp8" (E4M3 with one per-tensor scale, as vLLM does)
    axis:  "token" → one scale per token, shared by `group` consecutive elements of its head vector
           "channel" → one scale per channel, shared by `group` consecutive tokens (KIVI's keys)
    """

    bits: int = 8
    kind: str = "int"
    axis: str = "token"
    group: int = 128
    symmetric: bool = False

    def bits_per_element(self) -> float:
        if self.kind == "fp8":
            return 8.0  # one scale for the whole tensor: negligible
        return self.bits + SCALE_BITS * (1 + (not self.symmetric)) / self.group


@dataclass(frozen=True)
class KVSpec:
    """A KV-cache policy: how keys and values are stored (None = BF16), and what is kept.

    rotate_keys: rotate each key head vector by a random Hadamard matrix before quantizing (QuaRot-style)
    sinks, window: StreamingLLM eviction: keep the first `sinks` tokens and the last `window` ones only
    """

    name: str
    keys: TensorQuant | None = None
    values: TensorQuant | None = None
    rotate_keys: bool = False
    sinks: int | None = None
    window: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def bits_per_element(self) -> float:
        """Average over K and V."""
        k = self.keys.bits_per_element() if self.keys else 16.0
        v = self.values.bits_per_element() if self.values else 16.0
        return (k + v) / 2

    def kept_tokens(self, context: int) -> int:
        """Tokens the cache holds for a sequence of `context` tokens."""
        if self.window is None:
            return context
        return min(context, (self.sinks or 0) + self.window)


BF16 = KVSpec("BF16")


def bytes_per_token(cfg: Any, spec: KVSpec = BF16) -> float:
    """K and V for one token across all layers and KV heads, in the spec's storage format."""
    elements = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim
    return elements * spec.bits_per_element() / 8


def bytes_per_sequence(cfg: Any, context: int, spec: KVSpec = BF16) -> float:
    return spec.kept_tokens(context) * bytes_per_token(cfg, spec)


def max_tokens(cfg: Any, memory_bytes: float, spec: KVSpec = BF16) -> int:
    """How many tokens a KV budget of `memory_bytes` holds."""
    return int(memory_bytes // bytes_per_token(cfg, spec))


def max_sequences(cfg: Any, memory_bytes: float, context: int, spec: KVSpec = BF16) -> int:
    """How many sequences of `context` tokens fit at once: the concurrency ceiling for long requests."""
    return int(memory_bytes // bytes_per_sequence(cfg, context, spec))


def gib(n: float) -> float:
    return n / 2**30
