"""An analytical model of a serving step: what a configuration should cost, from bytes and FLOPs.

Everything is built from three kinds of numbers:

- **the model's sizes**, read from config.json (parameters, layers, heads);
- **the hardware's ceilings**, measured in M0 (memory bandwidth, peak FLOP/s per matmul format);
- **a few calibrated constants** (`Calibration`), each fitted on a named set of earlier measurements and
  listed with it in docs/07-performance-model.md. They cover what bytes and FLOPs cannot: fixed overheads,
  and how close each kernel gets to the ceilings.

The model, per decode step of `batch` sequences at mean context `context`:

    weights   bytes / bandwidth + FLOPs / peak        read every layer and the LM head once; multiply for
                                                      every token in the pass. Summed, not max'ed: near the
                                                      ridge a kernel reaches neither ceiling (smooth roofline)
    KV cache  batch × context × bytes per token / (bandwidth × efficiency of the attention kernel)
    overhead  fixed per step + per sequence (scheduling, sampling, streaming)

Prefill is compute-bound: linear FLOPs (2 × parameters per token) and attention FLOPs (quadratic in length).

A closed loop of N users then spends GPU time on prefills and on decode steps shared by the batch:

    time per request = prefill(uncached prompt) + output tokens × time per token / batch
    throughput       = output tokens / time per request          (output tokens per second, all users)
    TPOT             = time per token + the other users' prefills that interrupt it

Speculative decoding replaces "time per token" with a pass that verifies k + 1 tokens and keeps E of them:
(step with k + 1 tokens per sequence + k drafted tokens) / E. Prefix caching shrinks the uncached prompt.
A smaller KV format shrinks the KV term and, when the cache is the limit, raises the batch.

Stdlib only, so the laptop, the report and the dashboard can all run it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

BYTES_PER_WEIGHT = {"bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 4.125 / 8}  # INT4: + a scale per 128
KV_BYTES_PER_ELEMENT = {"bf16": 2.0, "fp8": 1.0}
FLASH_ATTN, FLASHINFER = "flash_attn", "flashinfer"


@dataclass(frozen=True)
class Hardware:
    """M0's measured ceilings."""

    bandwidth: float  # bytes/s
    peak: dict[str, float]  # FLOP/s per matmul format: "bf16", "fp8", "int8"
    l2_bytes: float = 0.0  # the GPU's L2 cache: what is read twice within it is read from memory once

    def matmul_peak(self, weights: str) -> float:
        """INT4 weights are dequantized to BF16 before the multiply (Marlin), so they use BF16's peak."""
        return self.peak["bf16" if weights == "int4" else weights]


@dataclass(frozen=True)
class Speculation:
    k: int  # tokens drafted per pass
    tokens_per_pass: float  # E: tokens kept per target pass, from measured acceptance
    draft_bytes: float  # what the drafter reads per drafted token (its layer and its LM head)


@dataclass(frozen=True)
class Stack:
    """One serving configuration."""

    weights: str = "bf16"  # "bf16" | "fp8" | "int8" | "int4"
    kv: str = "bf16"  # "bf16" | "fp8"
    backend: str = FLASH_ATTN  # FP8 KV runs on FlashInfer on this GPU
    prefix_caching: bool = False
    speculation: Speculation | None = None

    def __post_init__(self) -> None:
        if self.kv == "fp8" and self.backend != FLASHINFER:
            raise ValueError("an FP8 KV cache is served by FlashInfer here: set backend=FLASHINFER")


@dataclass(frozen=True)
class Load:
    """A closed-loop workload, as averages."""

    users: float  # sequences in flight on average (a closed loop of N users keeps a little under N)
    prompt_len: float
    output_len: float
    cached_len: float = 0.0  # prompt tokens a prefix cache would already hold
    context: float | None = None  # mean context of a decoding sequence, if known better than prompt + out/2
    shared_len: float = 0.0  # leading tokens that `sharers` sequences have in common (a system prompt)
    sharers: float = 1.0  # how many running sequences share each such prefix
    kv_tokens: float | None = None  # the cache's capacity in tokens, if it can limit the batch
    max_batch: int = 256  # vLLM's max_num_seqs


@dataclass(frozen=True)
class Calibration:
    """The constants bytes and FLOPs cannot give. Each is fitted on earlier measurements (report/m8.py)."""

    step_s: float = 0.0  # fixed cost of a decode step
    per_seq_s: float = 0.0  # added per sequence in the batch
    act_quant_s: float = 0.0  # per step, W8A8 only: quantizing each linear layer's input
    flash_attn_batch_penalty: float = 0.0  # FlashAttention's efficiency is 1 / (1 + penalty × log2(batch))
    flashinfer_efficiency: dict[str, float] = field(default_factory=lambda: {"bf16": 1.0, "fp8": 1.0})
    prefill_linear: dict[str, float] = field(default_factory=dict)  # share of peak FLOP/s, per weight format
    prefill_attention: float = 1.0  # share of BF16's peak FLOP/s reached by prefill attention
    draft_step_s: float = 0.0  # fixed cost per drafted token
    draft_per_seq_s: float = 0.0  # per drafted token and sequence: sampling and acceptance
    kv_slack: float = 1.0  # cache tokens a running sequence holds ÷ (prompt + output)
    request_s: float = 0.0  # per request, before its prefill starts: HTTP, tokenizing, scheduling


def linear_params(cfg: Any) -> int:
    """Parameters in the decoder layers: everything except the embedding / LM-head matrix."""
    matrix = cfg.vocab_size * cfg.hidden_size
    return cfg.num_params() - matrix * (1 if cfg.tie_word_embeddings else 2)


def weight_bytes(cfg: Any, weights: str) -> float:
    """Bytes a decode step streams: the quantized layers, and the LM head, which stays BF16."""
    return linear_params(cfg) * BYTES_PER_WEIGHT[weights] + cfg.vocab_size * cfg.hidden_size * 2


def weight_time(cfg: Any, hw: Hardware, stack: Stack, tokens: float) -> float:
    """Seconds to read the weights once and multiply them for `tokens` token positions."""
    head = cfg.vocab_size * cfg.hidden_size
    compute = 2 * linear_params(cfg) * tokens / hw.matmul_peak(stack.weights)
    compute += 2 * head * tokens / hw.peak["bf16"]
    return weight_bytes(cfg, stack.weights) / hw.bandwidth + compute


def attention_efficiency(stack: Stack, cal: Calibration, batch: float) -> float:
    """Share of the memory bandwidth at which the attention kernel reads the KV cache."""
    if stack.backend == FLASHINFER:
        return cal.flashinfer_efficiency[stack.kv]
    return 1 / (1 + cal.flash_attn_batch_penalty * math.log2(max(batch, 1)))


def kv_tokens_read(
    cfg: Any, hw: Hardware, stack: Stack, batch: float, context: float, load: Load | None
) -> float:
    """Cached tokens that must come from memory in one step.

    Normally every sequence's whole context. With prefix caching, sequences that share a prefix point at
    the *same* cache blocks, and attention runs layer by layer: if one layer's slice of the shared prefix
    fits in L2, the first sequence's read leaves it there for the others. The shared part then costs one
    read per group of sharers instead of one per sequence.
    """
    if load is None or not stack.prefix_caching or load.sharers <= 1 or load.shared_len <= 0:
        return batch * context
    per_layer = load.shared_len * cfg.kv_bytes_per_token(KV_BYTES_PER_ELEMENT[stack.kv]) / cfg.num_layers
    if per_layer > hw.l2_bytes:
        return batch * context
    shared = min(load.shared_len, context)
    return batch * (context - shared) + batch / min(load.sharers, batch) * shared


def kv_time(
    cfg: Any,
    hw: Hardware,
    stack: Stack,
    cal: Calibration,
    batch: float,
    context: float,
    load: Load | None = None,
) -> float:
    """Seconds to read the running sequences' keys and values once."""
    tokens = kv_tokens_read(cfg, hw, stack, batch, context, load)
    nbytes = tokens * cfg.kv_bytes_per_token(KV_BYTES_PER_ELEMENT[stack.kv])
    return nbytes / (hw.bandwidth * attention_efficiency(stack, cal, batch))


def step_time(
    cfg: Any,
    hw: Hardware,
    stack: Stack,
    cal: Calibration,
    batch: float,
    context: float,
    new_tokens: int = 1,
    load: Load | None = None,
) -> float:
    """Seconds for one decode pass: `batch` sequences, each adding `new_tokens` positions."""
    overhead = cal.step_s + cal.per_seq_s * batch
    if stack.weights in ("fp8", "int8"):
        overhead += cal.act_quant_s
    weights = weight_time(cfg, hw, stack, batch * new_tokens)
    return overhead + weights + kv_time(cfg, hw, stack, cal, batch, context, load)


def prefill_flops(cfg: Any, new_tokens: float, cached: float = 0.0) -> tuple[float, float]:
    """(linear, attention) FLOPs to prefill `new_tokens` after `cached` tokens already in the cache.

    Attention: every new token attends to the cached tokens and, causally, to half of the new ones; QKᵀ and
    PV each cost 2 FLOPs per (query, key, channel).
    """
    linear = 2 * linear_params(cfg) * new_tokens
    pairs = new_tokens * (cached + new_tokens / 2)
    return linear, 4 * pairs * cfg.head_dim * cfg.num_heads * cfg.num_layers


def prefill_time(
    cfg: Any,
    hw: Hardware,
    stack: Stack,
    cal: Calibration,
    new_tokens: float,
    cached: float = 0.0,
    own_pass: bool = True,
) -> float:
    """Seconds a prefill adds: its FLOPs, plus one read of the weights if it runs as a pass of its own.

    With other sequences decoding, vLLM puts the prompt's tokens into the same pass as their next step: the
    weights are streamed once for everyone, so the prefill only adds math (`own_pass=False`).
    """
    if new_tokens <= 0:
        return 0.0
    linear, attention = prefill_flops(cfg, new_tokens, cached)
    seconds = linear / (hw.matmul_peak(stack.weights) * cal.prefill_linear.get(stack.weights, 1.0))
    seconds += attention / (hw.peak["bf16"] * cal.prefill_attention)
    return seconds + (weight_bytes(cfg, stack.weights) / hw.bandwidth if own_pass else 0.0)


def time_per_token(
    cfg: Any,
    hw: Hardware,
    stack: Stack,
    cal: Calibration,
    batch: float,
    context: float,
    load: Load | None = None,
) -> float:
    """Seconds of GPU time per output token *of each sequence* in the batch (one step serves them all)."""
    spec = stack.speculation
    if spec is None:
        return step_time(cfg, hw, stack, cal, batch, context, load=load)
    verify = step_time(cfg, hw, stack, cal, batch, context, new_tokens=spec.k + 1, load=load)
    draft = spec.k * (cal.draft_step_s + spec.draft_bytes / hw.bandwidth + cal.draft_per_seq_s * batch)
    return (verify + draft) / spec.tokens_per_pass


def running_batch(load: Load, cal: Calibration) -> float:
    """Sequences decoding at once: the users, unless vLLM's limit or the KV cache holds fewer."""
    batch = float(min(load.users, load.max_batch))
    if load.kv_tokens is not None:
        batch = min(batch, load.kv_tokens / ((load.prompt_len + load.output_len) * cal.kv_slack))
    return max(batch, 1.0)


def predict(cfg: Any, hw: Hardware, stack: Stack, cal: Calibration, load: Load) -> dict[str, float]:
    """Throughput, TPOT and (for one user) TTFT of a closed-loop workload on one configuration."""
    batch = running_batch(load, cal)
    # The mean context while a sequence decodes. With mixed lengths the caller knows better: long requests
    # stay in the batch longest (report/m2.token_weighted_context).
    context = load.context if load.context is not None else load.prompt_len + load.output_len / 2
    cached = min(load.cached_len, load.prompt_len) if stack.prefix_caching else 0.0
    prefill = prefill_time(cfg, hw, stack, cal, load.prompt_len - cached, cached, own_pass=batch <= 1)
    per_token = time_per_token(cfg, hw, stack, cal, batch, context, load)
    request = prefill + load.output_len * per_token / batch
    if load.users <= 1:
        request += cal.request_s  # with one user the GPU idles while the next request is being handled
    tok_s = load.output_len / request
    # Between two of a user's tokens: one pass, plus the other users' prefills that cut in line.
    tpot = per_token + prefill * (batch - 1) / load.output_len
    return {
        "batch": batch,
        "tok_s": tok_s,
        "tpot_ms": 1e3 * tpot,
        "ttft_ms": 1e3 * (cal.request_s + prefill + per_token) if load.users <= 1 else float("nan"),
        "step_ms": 1e3 * step_time(cfg, hw, stack, cal, batch, context, load=load),
        "prefill_ms": 1e3 * prefill,
        "kv_share": kv_time(cfg, hw, stack, cal, batch, context, load)
        / step_time(cfg, hw, stack, cal, batch, context, load=load),
    }


def dollars_per_million(tok_s: float, dollars_per_hour: float) -> float:
    return dollars_per_hour / (tok_s * 3600) * 1e6
