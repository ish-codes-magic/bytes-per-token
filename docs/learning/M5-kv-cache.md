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

### Prediction vs measurement

<!-- BEGIN GENERATED: m5_predictions -->
*Predictions written in commit `b93afc6`, before the first measurement.*

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| vLLM KV-cache tokens, FP8 KV / BF16 KV, Qwen3-0.6B (x) | 1.9 – 2.02 | 1.91 | within range |
| vLLM KV-cache tokens, FP8 KV / BF16 KV, Qwen3-1.7B (x) | 1.9 – 2.02 | 1.83 | below range |
| Sequences running, capacity workload, FP8 KV / BF16 KV, Qwen3-0.6B (x) | 1.7 – 2.1 | 1.8 | within range |
| Output tokens/s, capacity workload, FP8 KV / BF16 KV, Qwen3-0.6B (x) | 1.3 – 1.9 | 1.77 | within range |
| Output tokens/s, capacity workload, FP8 KV / BF16 KV, Qwen3-1.7B (x) | 1.2 – 1.7 | 1.56 | within range |
| Saturated output tokens/s, FP8 KV / BF16 KV, Qwen3-0.6B (x) | 1.1 – 1.5 | 2.19 | above range |
| Saturated output tokens/s, FP8 KV / BF16 KV, Qwen3-1.7B (x) | 1.1 – 1.6 | 1.68 | above range |
| Saturated output tokens/s, BF16 KV on FlashInfer / on FlashAttention, Qwen3-0.6B (x) | 0.85 – 1.15 | 1.45 | above range |
| TPOT at 32k context, BF16 KV / FP8 KV, Qwen3-0.6B (x) | 1.2 – 1.7 | 1.47 | within range |
| TPOT at 32k context, BF16 KV / FP8 KV, Qwen3-1.7B (x) | 1.1 – 1.45 | 1.26 | within range |
| TTFT at 32k context, FP8 KV / BF16 KV, Qwen3-0.6B (x) | 0.85 – 1.3 | 0.99 | within range |
| KL vs BF16 cache, FP8 E4M3 KV (scale 1.0), Qwen3-0.6B | 0.002 – 0.04 | 0.0149 | within range |
| KL vs BF16 cache, INT8 KV per token, Qwen3-0.6B | 0.0002 – 0.003 | 0.0189 | above range |
| KL vs BF16 cache, INT4 KV, keys per token, Qwen3-0.6B | 0.05 – 1 | 6.02 | above range |
| KL ratio, INT4 rotated keys / INT4 per-token keys, Qwen3-0.6B (x) | 0.1 – 0.6 | 0.172 | within range |
| KL ratio, INT4 KIVI (keys per channel) / INT4 per-token keys, Qwen3-0.6B (x) | 0.05 – 0.5 | 0.00534 | below range |
| KL vs BF16 cache, INT2 KIVI, Qwen3-0.6B | 0.3 – 3 | 0.741 | within range |
| KL vs BF16 cache, StreamingLLM keeping 1,024 tokens, 2,048-token windows, Qwen3-0.6B | 0.05 – 0.5 | 0.0377 | below range |
| KL ratio, INT4 KIVI on Qwen3-1.7B / on Qwen3-0.6B (x) | 0.5 – 1.2 | 0.86 | within range |
| Median over layers of the largest/median channel ratio, keys ÷ values, Qwen3-0.6B (x) | 1.5 – 10 | 4.05 | within range |
| Needle pass rate, BF16 cache in nanoserve, Qwen3-0.6B (vLLM in M2: 1.0) | 0.9 – 1 | 1 | within range |
| Needle pass rate, FP8 KV in nanoserve, Qwen3-0.6B | 0.95 – 1 | 1 | within range |
| Needle pass rate, INT4 per-token keys, Qwen3-0.6B | 0 – 0.6 | 0 | within range |
| Needle pass rate, INT4 KIVI, Qwen3-0.6B | 0.85 – 1 | 1 | within range |
| Needle pass rate, StreamingLLM keeping 1,024 tokens, Qwen3-0.6B | 0.2 – 0.4 | 0.32 | within range |
| Needle pass rate, vLLM with FP8 KV, Qwen3-0.6B (BF16 in M2: 1.0) | 0.95 – 1 | 1 | within range |
| WikiText-2 perplexity in vLLM, FP8 KV / BF16 KV, Qwen3-0.6B (x) | 1 – 1.03 | 1.01 | within range |
| Share of multi-turn prompt tokens vLLM served from its prefix cache | 0.8 – 0.86 | 0.853 | within range |
| TTFT p50, multi-turn workload, prefix caching on / off, Qwen3-0.6B (x) | 0.15 – 0.6 | 0.272 | within range |
| Output tokens/s, multi-turn workload, prefix caching on / off, Qwen3-0.6B (x) | 1.1 – 1.6 | 1.68 | above range |
| Output tokens/s, multi-turn workload, prefix caching on / off, Qwen3-1.7B (x) | 1.15 – 1.8 | 1.71 | within range |
<!-- END GENERATED: m5_predictions -->

Capacity, long-context latency, needle recall and prefix caching came out as predicted. The misses trace to
three things I didn't know when I wrote the predictions:
1. how extreme Qwen3's key outliers are
2. how much the attention kernel matters on a saturated server
3. how little WikiText depends on distant context

Each is explained below.

### Keys have outlier channels; values don't

![Key and value channels](../../results/figures/m5_key_value_channels.png)
<!-- BEGIN GENERATED: caption-m5_key_value_channels -->
*Keys have outlier channels, values don't: in the median layer the largest key channel is 8.3× the median one, against 2.1× for values, so one scale per token wastes the grid on keys; layer 0's largest key reaches |506| (FP8 E4M3 tops out at 448).*
<!-- END GENERATED: caption-m5_key_value_channels -->

The early layers are extreme. On Qwen3-0.6B, layer 0's largest key goes past FP8 E4M3's largest value (448),
so at vLLM's default scale of 1.0 that channel saturates. Everything about quantizing this cache follows from
that picture:

<!-- BEGIN GENERATED: m5_policies -->
| KV policy | Bits per element | KiB per token | 32k-token sequences that fit | KL, Qwen3-0.6B | KL, Qwen3-1.7B | Top-1 agreement, 0.6B | Needle pass rate, 0.6B |
|---|---|---|---|---|---|---|---|
| BF16 (control) | 16.00 | 112.0 | 5 | 0.0000 | 0.0000 | 100.0% | 100% |
| FP8 E4M3, scale 1.0 | 8.00 | 56.0 | 10 | 0.0149 | 0.0120 | 93.7% | 100% |
| INT8, per token | 8.25 | 57.8 | 10 | 0.0189 | 0.0108 | 93.9% | 100% |
| INT8, keys per channel | 8.62 | 60.4 | 10 | 0.0014 | 0.0013 | 98.0% | 100% |
| INT4, per token | 4.25 | 29.8 | 20 | 6.0242 | 6.2818 | 10.3% | 0% |
| INT4, rotated keys | 4.25 | 29.8 | 20 | 1.0349 | 0.3401 | 58.0% | 51% |
| INT4 KIVI (keys per channel) | 4.62 | 32.4 | 18 | 0.0322 | 0.0277 | 91.1% | 100% |
| INT2 KIVI | 3.00 | 21.0 | 28 | 0.7410 | 0.5789 | 61.4% | 48% |
| StreamingLLM, 1,024 kept | 16.00 | 112.0 | 168 | 0.0377 | 0.0324 | 95.3% | 32% |
<!-- END GENERATED: m5_policies -->

- **A per-token grid fails on the keys, even at 8 bits.** One channel in the hundreds stretches each token's
  INT8 grid to steps of about 2, while most channels sit near 5. INT8 per token costs *more* KL than FP8 (whose
  floating-point grid keeps its relative precision per element). At 4 bits per token the model is destroyed
  (top-1 agreement 10%).
- **Per-channel keys fix it.** Added as a control once the per-token result came in: INT8 with per-channel
  keys cuts the KL by an order of magnitude. INT4 KIVI keeps every needle at under 5 bits per element, with a
  KL only a little above FP8's.
- **Rotation helps per-token keys, but not enough.** A Hadamard rotation spreads the outlier over all 128
  channels. But when one channel holds most of a vector's energy, every rotated channel inherits a large
  share of it, and a 15-level grid is still coarse for the rest. Rotation helps Qwen3-1.7B more than 0.6B,
  whose layer 0 is the extreme one.
- **KL and the needle measure different damage.** StreamingLLM's KL on 2,048-token windows is small:
  WikiText rarely needs anything over 1,000 tokens back. Its needle score, though, is exactly the cells
  where the needle sits inside the kept window. Eviction is invisible to perplexity-style metrics and fatal
  for recall.

![Needle grids](../../results/figures/m5_needle.png)
<!-- BEGIN GENERATED: caption-m5_needle -->
*Every policy keeps the needle except INT4, per token (0%), INT4, rotated keys (51%), INT2 KIVI (48%), StreamingLLM, 1,024 kept (32%).*
<!-- END GENERATED: caption-m5_needle -->

Production confirms the reference: vLLM's real FP8 cache keeps every needle, and its perplexity moves as
little as nanoserve's KL suggests:

<!-- BEGIN GENERATED: m5_vllm_quality -->
| Model | Perplexity, BF16 KV | Perplexity, FP8 KV | Change | Needle, BF16 KV | Needle, FP8 KV |
|---|---|---|---|---|---|
| Qwen3-0.6B | 19.54 | 19.78 | 1.21% | 100% | 100% |
| Qwen3-1.7B | 15.55 | 15.23 | -2.05% | 100% | 100% |
<!-- END GENERATED: m5_vllm_quality -->

Qwen3-1.7B's perplexity *improves* with an FP8 cache while its KL shows real change. That's the trap M3 found
with INT8: perplexity can reward damage, and KL is the honest measure.

### Capacity: twice the sequences in the same memory

![Concurrency vs context](../../results/figures/m5_concurrency.png)
<!-- BEGIN GENERATED: caption-m5_concurrency -->
*In the L4's 18.4 GiB KV budget, FP8 fits twice the sequences of BF16 and INT4 KIVI 3.5×; only eviction keeps the count flat as contexts grow. With 4k-token prompts vLLM ran 39 (BF16) vs 70 (FP8).*
<!-- END GENERATED: caption-m5_concurrency -->

<!-- BEGIN GENERATED: m5_kv_serving -->
| Model | Server | Attention | KV cache (tokens) | Running, capacity workload | Tokens/s, capacity | Tokens/s, saturated | TPOT at 32k (ms) | TTFT at 32k (ms) | $ / 1M tokens, saturated |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 KV | FLASH_ATTN | 172,640 | 39 | 327 | 1,853 | 20.26 | 3,333 | 0.120 |
| Qwen3-0.6B | BF16 KV, FlashInfer | FLASHINFER | 164,944 | 37 | 384 | 2,679 | 20.60 | 3,152 | 0.083 |
| Qwen3-0.6B | FP8 KV | FLASHINFER | 329,888 | 70 | 579 | 4,064 | 13.79 | 3,301 | 0.055 |
| Qwen3-0.6B | FP8 weights + FP8 KV | FLASHINFER | 342,096 | 72 | 594 | 3,567 | 12.87 | 3,219 | 0.062 |
| Qwen3-1.7B | BF16 KV | FLASH_ATTN | 152,800 | 35 | 255 | 1,591 | 29.02 | 4,593 | 0.140 |
| Qwen3-1.7B | BF16 KV, FlashInfer | FLASHINFER | 139,648 | 32 | 280 | 1,976 | 29.54 | 4,393 | 0.112 |
| Qwen3-1.7B | FP8 KV | FLASHINFER | 279,296 | 61 | 397 | 2,665 | 22.95 | 4,628 | 0.083 |
| Qwen3-1.7B | FP8 weights + FP8 KV | FLASHINFER | 312,752 | 67 | 437 | 2,882 | 18.38 | 4,109 | 0.077 |
<!-- END GENERATED: m5_kv_serving -->

- **The sizing model predicts vLLM's capacity.** vLLM's KV cache in tokens is its KV memory ÷ 112 KiB (BF16)
  or 56 KiB (FP8), and the measured points sit on the lines.
- **FlashInfer reserves more memory than FlashAttention.** Its BF16 cache holds fewer tokens, which is why
  FP8's token count is exactly 2× FlashInfer's BF16 but less than 2× FlashAttention's (the 1.7B miss).
- **With 96 users and 4k-token prompts, the cache decides the batch.** BF16 runs about 39 sequences and FP8
  about 70: nearly 2×, and throughput follows.
- **One user at 32k tokens:** decode reads ~3.7 GB of BF16 KV per step, so FP8 cuts time per output token by
  about the predicted amount. Time to first token doesn't move: prefill is compute.

### The saturated server: the kernel matters as much as the bytes

<!-- BEGIN GENERATED: m5_saturation -->
| Model | Server | Running | KV cache used | Step (ms) | GB read per step | Memory-bound share of the step | Prompt tokens per step | Preemptions | Output tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 KV | 252 | 92% | 136 | 19.3 | 54% | 433 | 36 | 1,853 |
| Qwen3-0.6B | BF16 KV, FlashInfer | 247 | 94% | 92 | 19.0 | 79% | 424 | 226 | 2,679 |
| Qwen3-0.6B | FP8 KV | 252 | 48% | 62 | 10.3 | 63% | 437 | 0 | 4,064 |
| Qwen3-0.6B | FP8 weights + FP8 KV | 251 | 46% | 71 | 9.8 | 53% | 438 | 0 | 3,567 |
| Qwen3-1.7B | BF16 KV | 233 | 96% | 146 | 20.3 | 53% | 396 | 350 | 1,591 |
| Qwen3-1.7B | BF16 KV, FlashInfer | 214 | 97% | 108 | 19.0 | 67% | 357 | 439 | 1,976 |
| Qwen3-1.7B | FP8 KV | 252 | 57% | 94 | 12.5 | 51% | 437 | 0 | 2,665 |
| Qwen3-1.7B | FP8 weights + FP8 KV | 251 | 51% | 87 | 11.1 | 48% | 438 | 0 | 2,882 |
<!-- END GENERATED: m5_saturation -->

The "memory-bound share" column divides the time streaming each step's bytes would take (weights plus the KV
in use, at M0's bandwidth) by the measured step:

- **On FlashAttention (vLLM's default here), saturated BF16 steps reach about half of memory speed.** That's
  M2's 54%. M2 blamed the chunked-prefill tokens mixed into every step. But the same mixed steps on
  FlashInfer, with the same ~250 sequences and a full cache, reach far more. A large part of M2's gap is how
  FlashAttention handles mixed prefill/decode batches on this GPU, not the mixing itself. Why exactly is a job
  for M7's profiler.
- **FP8 KV then halves the bytes.** The cache is half as full for the same sequences, and the preemptions
  vanish. The step shrinks less than the bytes: what remains (prefill compute, launches) is fixed.
- **Credited honestly, the gain splits in two**: the FlashInfer kernel, and FP8 storage on the same kernel.
  Waterfall v2 below shows them as separate steps. FP8 storage alone, on the same kernel, lands near the
  predicted range; the 2.2× total doesn't.
- **FP8 weights on top make the small model's saturated server slower.** W8A8's activation-quantization pass
  (M4) costs more than halving 1.2 GB of weights saves when 250 sequences share each step. On 1.7B, with
  three times the weights, it still pays.

### Prefix caching

![Radix tree of the multi-turn workload](../../results/figures/m5_radix_tree.png)
<!-- BEGIN GENERATED: caption-m5_radix_tree -->
*128 prompts (271,159 tokens) collapse into 39,235 distinct tokens: each app's system prompt is computed once, and 86% of all prompt tokens could come from the cache.*
<!-- END GENERATED: caption-m5_radix_tree -->

The reference radix tree ([`kv/prefix_cache.py`](../../src/fastserve/kv/prefix_cache.py)) replays the
workload and predicts the cache's best case. vLLM's block-hash cache gets within a hair of it: it can only
match whole 16-token blocks, and loses a few tokens at each boundary.

<!-- BEGIN GENERATED: m5_prefix -->
| Model | Workload | Prefix caching | Prompt tokens cached | TTFT p50 (ms) | TPOT p50 (ms) | Tokens/s | $ / 1M tokens |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | multi-turn | off | 0.0% | 222 | 18.15 | 376 | 0.591 |
| Qwen3-0.6B | multi-turn | on | 85.3% | 60 | 11.93 | 631 | 0.352 |
| Qwen3-0.6B | shared-prefix | off | 0.0% | 274 | 17.78 | 366 | 0.607 |
| Qwen3-0.6B | shared-prefix | on | 95.9% | 40 | 9.24 | 810 | 0.274 |
| Qwen3-1.7B | multi-turn | off | 0.0% | 479 | 32.93 | 199 | 1.114 |
| Qwen3-1.7B | multi-turn | on | 85.3% | 135 | 21.59 | 341 | 0.652 |
| Qwen3-1.7B | shared-prefix | off | 0.0% | 591 | 31.88 | 196 | 1.132 |
| Qwen3-1.7B | shared-prefix | on | 95.9% | 68 | 18.70 | 402 | 0.552 |
<!-- END GENERATED: m5_prefix -->

![TTFT vs cached share](../../results/figures/m5_prefix_ttft.png)
<!-- BEGIN GENERATED: caption-m5_prefix_ttft -->
*Prefix caching on the multi-turn workload: Qwen3-0.6B: TTFT p50 222 → 60 ms, prefill 85% smaller; Qwen3-1.7B: TTFT p50 479 → 135 ms, prefill 85% smaller.*
<!-- END GENERATED: caption-m5_prefix_ttft -->

- **TTFT drops because most of each prompt is never computed again.** The few requests that found nothing
  cached (each app's first turn) sit at the left, with the slow uncached requests.
- **Throughput rose more than predicted, and TPOT fell too.** Without caching, every step that carries a
  2,000-token prefill chunk is a slow step for all 8 running sequences. With caching, those chunks shrink to
  a few hundred tokens, so decode steps stay fast. I predicted the prefill saving but not this second-order
  effect on everyone else's decode.

### Cost: waterfall v2

![Waterfall v2](../../results/figures/m5_waterfall.png)
<!-- BEGIN GENERATED: caption-m5_waterfall -->
*Waterfall v2, Qwen3-0.6B, the cheapest setup against BF16: saturated server -54% (FP8 KV; the FlashInfer kernel alone -31%); 96 users, 4k prompts -45% (FP8 weights + FP8 KV; the FlashInfer kernel alone -15%); multi-turn chat -40% (prefix caching).*
<!-- END GENERATED: caption-m5_waterfall -->

### Explaining every gap

| Observation | Explanation (lever) |
|---|---|
| INT8 per token costs more KL than FP8 | Qwen3's key outliers (past FP8's range in 0.6B's layer 0) stretch each token's integer grid; FP8 keeps relative precision per element. Per-channel keys fix it. |
| INT4 per token is catastrophic; KIVI is fine | The same outliers at 15 levels. One scale per channel isolates them (*move fewer bytes*, done right). |
| StreamingLLM's KL is small, its recall poor | WikiText needs little context beyond 1,000 tokens; the needle needs exactly what was evicted. |
| FP8 KV fits 1.83× (not 2×) the tokens on 1.7B | FP8 runs FlashInfer, which reserves more non-KV memory than FlashAttention. Against BF16 on FlashInfer it's exactly 2×. |
| Saturated throughput 2.2× (predicted ≤1.5×) | Two levers stacked: a better kernel for mixed batches (*less waste*) and half the KV bytes (*move fewer bytes*). The FlashInfer control separates them. |
| FP8 weights slow the saturated 0.6B server | W8A8's per-layer activation quantization (M4) outweighs the weight bytes saved when 250 sequences share a step. |
| Prefix caching raises throughput more than predicted | Shorter prefill chunks stop slowing the other sequences' decode steps (*less waste*), on top of the skipped prefill (*more tokens per byte*). |
| vLLM's cached share is just under the radix tree's | Block hashing matches whole 16-token blocks only. |
| FP8 KV improves Qwen3-1.7B's perplexity | Perplexity can reward damage (M3); KL shows the cache did change the model. |

### When each technique is worth it

- **FP8 KV cache: almost always, on this GPU.** It costs about as much KL as FP8 weights and no recall, and it
  doubles capacity. Check the model's key range against FP8's: Qwen3-0.6B's largest key saturates at the
  default scale, and a calibrated scale would avoid it.
- **INT8 / INT4 KV: only with per-channel keys** (KIVI-style). Per-token keys are wrong for models with key
  outliers, at any bit width. INT4 KIVI needs a kernel that dequantizes in attention, which vLLM 0.30 doesn't
  have for this format. That's M7's main kernel.
- **INT2 KV: not for retrieval.** It loses about half the needles.
- **Eviction (StreamingLLM): only when old context truly doesn't matter**, e.g. a bounded chat window. Never
  for documents, RAG or anything that asks about the middle of the context.
- **Prefix caching: whenever requests share prefixes**, which most chat, agent and RAG traffic does. It's
  lossless, outputs are unchanged (nanoserve's test checks exactly that), and its cost is cache memory that
  would otherwise sit unused. Its gain depends entirely on the traffic: none for unique prompts.

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
