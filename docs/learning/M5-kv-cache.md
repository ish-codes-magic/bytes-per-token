# M5: KV-cache engineering

M4 showed that shrinking the weights speeds up one user a lot and a busy server hardly at all, because at a
large batch most bytes in a decode step come from the **KV cache**, not the weights. M5 attacks the KV cache
directly, in three ways:

1. **Store it in fewer bits** (FP8, INT8, INT4, INT2): fewer bytes per token, so more tokens fit and each step
   reads less.
2. **Keep less of it** (StreamingLLM eviction): a fixed memory per sequence, at a price in recall.
3. **Compute it once** (prefix caching): requests that share a prefix share its keys and values.

Every lossy change is judged on quality *at long context*. That's where a damaged cache shows: a model can
look fine on short text and still lose the one fact it was asked about 20,000 tokens earlier.

## 1. Intuition

**What the cache is.** At every layer, each token's hidden state is projected into a key (what this token
offers) and a value (what it hands over when attended to). A new token's query compares itself with every
earlier key and takes a weighted mix of their values. Without a cache, every step would recompute all
earlier keys and values. With one, they're computed once and read back at every later step: memory traded
for compute.

**Why it dominates.** Weights are read once per step, however many sequences are in the batch. KV is read
once per step *per sequence*, and grows with each sequence's length. So:

- one user, short context: weights dominate (M1, M4)
- many users, or long contexts: the KV cache dominates (M2's saturated server read about 15× more KV than
  weight bytes)

The KV cache costs twice:
- **Capacity.** The cache's size caps how many sequences can run at once, and so the batch and the
  throughput.
- **Bandwidth.** Every decode step streams the whole cache of every running sequence.

**Why keys are harder to quantize than values.** In most LLMs, a few key *channels* (fixed dimensions of the
head vector) are large for nearly every token. RoPE rotates pairs of channels at fixed frequencies, and the
low-frequency pairs carry most of the magnitude. A per-token scale has to cover that outlier channel, so
every other channel of the token gets a coarse grid. Values have no such fixed structure.

KIVI's fix is to quantize keys **per channel** (one scale per channel per group of tokens) and values per
token. The rotation fix (QuaRot) spreads each outlier over all channels first, so a per-token grid fits
again.

**Eviction** keeps a fixed budget per sequence. StreamingLLM keeps the first few tokens (the *attention sinks*
from M1, which soak up attention mass) plus a recent window. Memory stays flat however long the text, but
anything outside the window is gone. Recall of a fact in the middle of a long document fails by design.

**Prefix caching** exploits causality. A token's key and value depend only on the tokens before it, so two
requests that start with the same 2,000 tokens compute identical keys and values for them. Cache them once,
and the second request prefills only what's new. SGLang organizes the cached prefixes as a radix tree. vLLM
hashes each 16-token block together with the hash of the block before it. Both find the longest cached
prefix.

## 2. The math

**Bytes per token** (Qwen3-0.6B and Qwen3-1.7B have the same attention shape: 28 layers, 8 KV heads,
head_dim 128):

```
KV bytes per token = 2 (K, V) × layers × kv_heads × head_dim × bytes per element
                   = 2 × 28 × 8 × 128 × 2 B = 114,688 B = 112 KiB  (BF16)
```

A quantized cache also stores its scales (and zero-points, for asymmetric grids):

```
bits per element = bits + 16 × (1 + has_zero_point) / elements_per_scale

FP8, one scale per tensor:                 8
INT8 per token (groups of 128):            8 + 32/128 = 8.25
INT4 per token:                            4.25
INT4 KIVI keys (per channel, 32 tokens):   4 + 32/32 = 5;   values per token: 4.25;   average 4.625
INT2 KIVI (keys and values, groups of 32): 2 + 1 = 3
```

**Capacity.** With M bytes for the KV cache (vLLM reports it at startup: about 18.4 GiB for Qwen3-0.6B on the
L4):

```
max tokens        = M / bytes per token          BF16: 18.44 GiB / 112 KiB ≈ 172,600
max sequences (L) = max tokens / L               L = 4,352 (4k prompt + 256 output): ≈ 39
StreamingLLM:       every sequence holds at most sinks + window tokens, whatever L is
```

**Bytes per decode step** (from M4), now with the KV format:

```
bytes/step = weight bytes + Σ_running (context_i) × KV bytes per token
```

Halving KV bytes per token does one of two things:
- **The batch is capped by capacity** (the cache is full): twice as many sequences fit, and each step reads
  about the *same* bytes for twice the tokens. Throughput rises toward 2× (*more tokens per byte*).
- **The batch is capped elsewhere** (vLLM's default `max_num_seqs` of 256, or the users simply aren't
  there): the same sequences run, and each step reads less. Throughput rises by the KV share of the step
  (*move fewer bytes*).

**Prefix caching** saves prefill compute and time to first token:

```
prefill FLOPs saved = 2 × params × cached tokens            (plus attention over them)
TTFT ≈ queueing + prefill time of the *uncached* tokens + one decode step
```

## 3. Setup

- **Reference (nanoserve, BF16 weights):** a KV policy rounds K and V the way a quantized cache would store
  them, after RoPE ([`kv/quant.py`](../../src/fastserve/kv/quant.py)). Attention then runs on exactly what
  the cache would hold. Eviction masks the slots the cache would have dropped.
  - Per-channel keys need whole groups of tokens. The last `T mod 32` tokens of each prefill chunk stay full
    precision (KIVI's "residual"), and so do decode tokens.
  - Quality has two measures:
    - **KL** against the BF16 cache on M2's WikiText-2 windows (2,048 tokens)
    - **M2's needle grid** (1k–32k tokens × 5 depths × 3 secrets), prefilled through a real cache in
      512-token chunks
- **Policies:**
  - BF16 (control)
  - FP8 E4M3 with scale 1.0, vLLM's default
  - INT8 per token
  - INT4 per token
  - INT4 with Hadamard-rotated keys
  - INT4 KIVI
  - INT2 KIVI
  - StreamingLLM: 4 sinks + 1,020 recent tokens
- **Production (vLLM 0.30.0 on the L4):** `--kv-cache-dtype fp8`. On the L4 that also switches the attention
  kernel to FlashInfer (FlashAttention's FP8 path needs Hopper's FA3), so a BF16 cache on FlashInfer is the
  control. FP8 weights + FP8 KV make the waterfall's next bar.
- **Prefix caching:** vLLM with automatic prefix caching on vs off, and per-request cached-token counts
  (`--enable-prompt-tokens-details`). Two workloads:
  - the **multi-turn** chat workload: 4 apps with 1,536-token system prompts, 32 conversations of 4 turns,
    each turn extending the previous one
  - M2's **shared-prefix** agent workload
- **Workloads for the cache's capacity:**
  - **capacity**: 96 users with 4k-token prompts, more than a BF16 cache holds
  - **saturation**: as in M2 and M4
  - **long_32k**: one user at 32k tokens, where decode reads ~3.6 GB of BF16 KV per step

## 4. Prediction (written before any M5 measurement)

| Quantity | Prediction | Reasoning |
|---|---|---|
| KV tokens, FP8 / BF16 | **1.9–2.02×** | Half the bytes per token. Slightly under 2× if FlashInfer reserves more workspace. |
| Running sequences, capacity workload, FP8 / BF16, 0.6B | **1.7–2.1×** | BF16 fits ~39 sequences of 4,352 tokens, FP8 ~79. Both are below the 96 users. |
| Throughput, capacity workload, 0.6B | **1.3–1.9×** | Each decode step reads the same full cache for 2× the tokens, but prefill work per request is unchanged. With ~0.12 s of prefill and ~0.46 s (BF16) vs ~0.23 s (FP8) of decode share per request: ≈1.65×. |
| Throughput, capacity workload, 1.7B | **1.2–1.7×** | Same mechanism, with 3.4 GB of weights in every step and 3× the prefill compute: ≈1.45×. |
| Saturated throughput, FP8 / BF16, 0.6B | **1.1–1.5×** | M4's saturated runs peaked at exactly 256 running: vLLM's `max_num_seqs` default, not the cache, caps the batch. FP8 halves the KV bytes (~74 → ~39 ms of streaming per step) but not M2's ~60 ms of prefill-interference overhead: ≈1.3×. |
| Saturated throughput, 1.7B | **1.1–1.6×** | The cache was full there (96%, 363 preemptions). FP8 lifts it to the 256 cap and halves its bytes: ≈1.35×. |
| BF16 KV on FlashInfer / on FlashAttention, saturated, 0.6B | **0.85–1.15×** | Two good decode kernels; the control should be close. |
| TPOT at 32k, BF16 / FP8, 0.6B | **1.2–1.7×** | One step reads 1.2 GB of weights + 3.7 GB of KV in BF16, + 1.8 GB in FP8: ≈1.55×. |
| TPOT at 32k, 1.7B | **1.1–1.45×** | 3.4 GB of weights dilute the KV saving: ≈1.3×. |
| TTFT at 32k, FP8 / BF16, 0.6B | **0.85–1.3×** | Prefill is compute: the linears don't change, and attention still runs in 16 bits after reading FP8. |
| KL, FP8 KV | **0.002–0.04** | E4M3's 3 mantissa bits: ~4% RMS error per element on K and V. FP8 activations in M3 cost ~0.02 KL. |
| KL, INT8 per token | **0.0002–0.003** | 255 levels per 128-element vector: tiny. |
| KL, INT4 per token | **0.05–1.0** | 15 levels per vector, stretched by the outlier key channels. |
| KL ratio, rotated keys / per-token | **0.1–0.6×** | The rotation spreads the outlier channels' energy over all 128 channels. |
| KL ratio, KIVI / per-token | **0.05–0.5×** | Per-channel scales isolate the outlier channels completely. |
| KL, INT2 KIVI | **0.3–3** | 3 levels: per-channel scales help, but 2 bits is very coarse. |
| KL, StreamingLLM (1,024 kept) on 2,048-token windows | **0.05–0.5** | Half the positions lose their distant context; WikiText's dependence on it is real but modest. |
| KL ratio, INT4 KIVI on 1.7B / 0.6B | **0.5–1.2×** | Same attention shape; the larger model is usually a little more robust. |
| Key ÷ value outlier ratio (largest/median channel) | **1.5–10×** | Keys carry fixed outlier channels (RoPE's low frequencies). Qwen3's QK-norm normalizes each head vector but its learned per-channel weight can still create them. |
| Needle pass rate, BF16 in nanoserve | **0.9–1.0** | vLLM passed every cell in M2. nanoserve computes the same function. |
| Needle, FP8 KV | **0.95–1.0** | A small KL; retrieval needs the right key to win, not exact values. |
| Needle, INT4 per token | **0.0–0.6** | At 32k, rounding noise in every key competes with the one key that matters. |
| Needle, INT4 KIVI | **0.85–1.0** | KIVI's papers report near-lossless retrieval at 4 bits. |
| Needle, StreamingLLM | **0.2–0.4** | It passes only when the needle sits in the last 1,020 tokens (depth 1.0) or the whole prompt nearly fits (the 1k cells). |
| Needle, vLLM FP8 KV, 0.6B | **0.95–1.0** | As for nanoserve's FP8. |
| Perplexity in vLLM, FP8 KV / BF16 | **1.002–1.03×** | Consistent with a KL around 0.01. |
| Prefix-cached share of multi-turn prompt tokens | **0.80–0.86** | The radix-tree replay of the workload reuses 85.5% of prompt tokens; vLLM matches whole 16-token blocks only. |
| TTFT p50, multi-turn, caching on / off, 0.6B | **0.15–0.6×** | ~2,100-token prompts become ~300 uncached tokens. |
| Throughput, multi-turn, on / off, 0.6B | **1.1–1.6×** | At 8 users decode (64 tokens each) is a large share of the work; prefill shrinks ~7×. |
| Throughput, multi-turn, on / off, 1.7B | **1.15–1.8×** | Prefill is 3× costlier per token, so saving it matters more. |

## 5. Result

*(Written after the runs.)*

## 6. Check your understanding

1. Derive the KV bytes per token of Qwen3-1.7B from its config. Why is it the same as Qwen3-0.6B's, though the
   model is three times larger?
   <details><summary>Answer</summary>2 × 28 layers × 8 KV heads × 128 head_dim × 2 bytes = 112 KiB. KV size
   depends only on the attention shape (layers, KV heads, head_dim). The larger model widens the hidden state
   and the MLP and adds query heads, but keeps the same 8 KV heads of 128 dimensions (grouped-query
   attention).</details>
2. Halving KV bytes per token sometimes doubles throughput and sometimes barely moves it. When is which?
   <details><summary>Answer</summary>When the cache is full and limits how many sequences run, halving its
   size doubles the batch: each step reads about the same bytes for twice the tokens. When something else caps
   the batch (vLLM's max_num_seqs, too few users), the same sequences run and each step just reads less KV.
   The gain is then the KV share of the step, which is small for one short request.</details>
3. Why are keys quantized per channel in KIVI, and values per token?
   <details><summary>Answer</summary>Keys have outlier *channels*: a few dimensions large for almost every
   token. A per-token scale must cover them, wasting the grid for the other channels. One scale per channel
   isolates them. Values have no fixed outlier channels, and each output mixes value vectors token by token,
   so a per-token grid fits them and keeps each token's error independent.</details>
4. StreamingLLM keeps memory constant. Why does it fail needle-in-a-haystack, and why does it need the "sink"
   tokens?
   <details><summary>Answer</summary>Anything outside the recent window is evicted, so a fact in the middle of
   a long document is simply no longer there to attend to. The sink tokens (the first few) receive a large
   share of attention in every layer, as a place to put attention mass. Evicting them shifts that mass onto
   other tokens and breaks the model's outputs, while keeping 4 of them costs almost nothing.</details>
5. A request has a 6,000-token system prompt shared with 1,000 other requests. What does prefix caching save,
   in FLOPs and in TTFT, on Qwen3-0.6B?
   <details><summary>Answer</summary>About 2 × 0.6e9 × 6,000 ≈ 7 TFLOP of linear-layer compute per request (plus
   attention over those tokens), so ≈7 PFLOP over the 1,000 requests. TTFT drops from the time to prefill
   6,000+ tokens (~0.2 s at ~40 TFLOP/s) to the time to prefill just the new suffix. vLLM points the request's
   block table at the cached blocks; nothing is copied.</details>

## 7. Further reading

- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, 2024.
- Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*, 2024.
- Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, 2024.
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), 2023.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention), 2024.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023.
- The vLLM documentation: *Automatic Prefix Caching* and *Quantized KV Cache*.
