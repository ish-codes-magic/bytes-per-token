# M8: The full stack

M4–M7 each added one technique and measured it alone. M8 turns them all on, and asks three questions:

1. **How much does each technique contribute when the others are on?** (the ablation)
2. **Do the gains multiply?** (the interactions)
3. **Could we have predicted all of it from bytes and FLOPs?** (the performance model)

The model is built and frozen *before* the full stack is measured. Its predictions for every M8 server are in
`benchmarks/predictions/m8_model.json`, committed before the first server started.

## 1. Intuition

**The relay team.** A decode step is a relay: read the weights, read the KV cache, do the math, pay the
overheads. Making one runner faster helps only as much as that runner's share of the lap. Halve the weights'
bytes and a lap that was 90% weights gets much shorter; a lap that was 80% KV cache barely notices.

**Why gains don't multiply.** Two techniques multiply only if they shorten *different* runners' legs, or the
same leg by independent factors. They *compete* when both spend the same slack:

- FP8 weights make the step shorter. Speculative decoding fills the step's idle compute with guesses. A
  shorter step has less idle compute per token saved, and the drafter's cost is a larger share of it.
- FP8 KV shrinks the cache read. Prefix caching makes sequences share cache blocks, which also shrinks what
  is read. The second one finds less to save.

They *help each other* when one removes a different bottleneck the other exposes: prefix caching removes
prefill time, which leaves a request that is all decode, and decode is where speculation works.

**Ablation, done properly.** Change one thing at a time, keep everything else fixed (workloads, hardware,
software), and never report a technique's gain without saying what else was on. A technique has two honest
numbers: what it adds to nothing (alone), and what the full stack loses without it (leave-one-out). When
techniques interact, the two differ.

**A model you can be wrong with.** A performance model is only worth something if it could have failed. So:
calibrate it on old data, write down what it says about new configurations, then measure them.

## 2. The math

### 2.1 Ladder, leave-one-out, and the grid

With four on/off techniques there are 2⁴ = 16 configurations. Measuring all 16 gives everything at once:

- the **ladder** (cumulative): base → w → wk → wkp → wkps;
- **leave-one-out**: wkps against kps, wps, wks, wkp;
- every **pair** against its two singles.

Letters: **w** FP8 weights, **k** FP8 KV cache, **p** prefix caching, **s** speculative decoding (an EAGLE-3
head, 3 drafted tokens). A server's label is the letters that are on.

### 2.2 Interaction

For techniques A and B on one workload, with speedups S over the base:

    interaction(A, B) = S(A + B) / (S(A) · S(B))

1 means the gains multiply. Below 1 they compete. Above 1 each makes the other more useful. Two repeated runs
of the same server say how far from 1 is noise.

### 2.3 The serving model

One decode step for `b` sequences at mean context `c`:

    t_step = t_fixed + b · t_seq                                   overheads
           + W / B + 2 · P · b / F                                 weights: bytes ÷ bandwidth + FLOPs ÷ peak
           + b · c · K / (B · η)                                   KV cache: bytes ÷ (bandwidth × efficiency)

- W: bytes of weights in the serving format (the LM head stays BF16). B: M0's read bandwidth.
- P: parameters; F: M0's peak FLOP/s for that format. K: KV bytes per token (halved by FP8).
- η: how close the attention kernel gets to the bandwidth (calibrated per kernel).
- The weights term **adds** memory time and compute time instead of taking the larger. Near the ridge point
  a kernel reaches neither ceiling, and the data fits the sum better than the max.

Prefill of q new tokens after c₀ cached ones is compute-bound:

    t_prefill = 2 · P · q / (F · η_lin) + 4 · q · (c₀ + q/2) · D · H · L / (F_bf16 · η_attn)

A closed loop of N users spends GPU time on prefills and on steps that the whole batch shares:

    time per request = t_prefill + O · t_token / b            O output tokens, b sequences in flight
    throughput       = O / time per request

Each technique changes one term:

| Technique | What changes in the model |
|---|---|
| FP8 / INT4 weights | W shrinks; F changes (FP8 math is faster, INT4 is dequantized to BF16) |
| FP8 KV cache | K halves; the attention kernel changes (η); if the cache limited the batch, b grows |
| Prefix caching | q shrinks to the uncached part of the prompt; shared blocks are read once per group (below) |
| Speculative decoding | t_token = (step with k + 1 tokens per sequence + k drafted tokens) / E |

E is the tokens kept per target pass (M6). A drafted token costs what the drafter reads: one decoder layer
and its reduced LM head.

**Shared prefixes in decode.** With prefix caching, sequences that share a prompt prefix point at the *same*
cache blocks. Attention runs layer by layer, and one layer's slice of a 1,500-token prefix is a few megabytes:
it fits in the GPU's L2 cache. So the first sequence's read leaves it there for the others, and the shared
part is read from memory once per group, not once per sequence. This came out of fitting the model: M5's
prefix-caching servers decoded faster than their prefill savings could explain. M8's runs with other caches
(FP8 KV halves the slice) test it.

### 2.4 Calibration: what is fitted, and on what

Bandwidth and peak FLOP/s come from M0. Sizes come from config.json. The rest are constants fitted on named
groups of M2–M6 measurements, and every other point is held out (section 3).

## 3. Setup

**The plan** ([config](../../benchmarks/configs/m8_ablation.yaml)): 30 vLLM servers on one L4, one container
each.

- Qwen3-1.7B: all 16 combinations of w, k, p, s; a FlashInfer control (`wf`: FP8 KV's attention kernel on a
  BF16 cache, to separate the kernel from the bytes); the INT4 branch (`a`, `akps`); and the base and the full
  stack run twice.
- Qwen3-0.6B: the ladder and leave-one-out, with a community EAGLE-3 head.

**Five workloads**, the same on every server:

| Workload | Users | Prompts | What it stresses |
|---|---|---|---|
| Latency | 1 | real: chat, code, math, summarization | the step itself |
| Busy | 64 | the same four tasks | a full batch: math and KV |
| Capacity | 96 | 4,096 random tokens | the KV cache's size |
| Multi-turn | 8 | conversations on 4 shared system prompts | prefill and shared prefixes |
| Long | 1 | 32,768 random tokens | the KV read (ladder servers only) |

Speculation is judged on real prompts where they exist. The three random-token workloads have no measured
acceptance rate, so the model borrows the real-prompt one: its weakest input.

**Quality** is measured for the lossy part of the stack (FP8 weights + FP8 KV): perplexity, needle recall and
the task suite, in vLLM. Prefix caching does not change outputs. Speculation is lossless in distribution (M6).

**Energy**: GPU power is sampled during every load, for tokens per joule.

**The custom kernels** (M7) are not in vLLM. Their step is reported as nanoserve's measurement plus the
projection from M7, labeled as such (the gate decision).

**A known bias in the inputs.** M0's cold matmul timings and M1's decode points were taken with the
write-flush that M7 found biased. The model does not use them: it takes M0's bandwidth (large transfers, no
flush) and peak FLOP/s (large matmuls), and vLLM's own measurements.

### The model, calibrated on M2–M6

<!-- BEGIN GENERATED: m8_calibration -->
| Constant | Value | Fitted on (points) |
|---|---|---|
| Fixed cost of a decode step | 1.21 ms | 10 (BF16, ≤ 16 users) |
| Added per sequence in the batch | 0 µs | 10 (BF16, ≥ 64 users: decode, capacity, saturation) |
| FlashAttention: efficiency lost per doubling of the batch | 0.049 | 10 (BF16, ≥ 64 users: decode, capacity, saturation) |
| W8A8's extra per step | 0.49 ms | 16 (FP8/INT8, ≤ 16 users) |
| FlashInfer's share of the bandwidth, BF16 cache | 95% | 6 (M5's FlashInfer control) |
| FlashInfer's share of the bandwidth, FP8 cache | 83% | 6 (M5's FP8 KV servers) |
| Prefill: share of BF16's peak FLOP/s, linear layers | 81% | 12 (one user, 8k–32k prompts: TTFT) |
| Prefill: share of BF16's peak FLOP/s, attention | 80% | the same |
| Prefill: share of its own peak FLOP/s, FP8 layers | 50% | 4 (one user, long prompts: TTFT) |
| Prefill: share of its own peak FLOP/s, INT4 layers | 86% | 8 (one user, long prompts: TTFT) |
| Prefill: share of its own peak FLOP/s, INT8 layers | 48% | 4 (one user, long prompts: TTFT) |
| Per request, before its prefill | 4.1 ms | 6 (one user: TTFT) |
| Fixed cost per drafted token | 0.26 ms | 4 (EAGLE-3, one user) |
| Per drafted token and sequence | 10 µs | 1 (EAGLE-3, 64 users) |
| Cache tokens a running sequence holds ÷ its length | 1.066 | the capacity runs' counters |
<!-- END GENERATED: m8_calibration -->

Some constants say something by themselves:

- **No per-sequence overhead was needed**, and FlashAttention loses a few percent of efficiency per doubling
  of the batch.
- **FP8 layers reach half of their peak FLOP/s in prefill.** FP8 math is twice as fast as BF16 on paper;
  in a real prefill it is barely faster.
- **A drafted token costs a fixed quarter of a millisecond** on top of the bytes the head reads.

### Does it reproduce what was already measured?

<!-- BEGIN GENERATED: m8_validation -->
| Measured in | Points | Count | Median error, tokens/s | Worst | Within 15% | Median error, TTFT (prompts ≥ 2,000 tokens) | Median error, TTFT (shorter prompts) |
|---|---|---|---|---|---|---|---|
| M2 baselines | used in a fit | 8 | 1.1% | 4% | 100% | 3.3% | 7 ms |
| M2 baselines | held out | 10 | 4.7% | 7% | 100% | — | 13 ms |
| M4 weight formats | used in a fit | 50 | 2.3% | 17% | 94% | 3.6% | 9 ms |
| M4 weight formats | held out | 40 | 5.0% | 20% | 90% | — | 6 ms |
| M5 KV cache and prefix caching | used in a fit | 18 | 2.7% | 14% | 100% | 3.2% | — |
| M5 KV cache and prefix caching | held out | 14 | 2.8% | 27% | 79% | 2.5% | — |
| M6 speculative decoding | used in a fit | 5 | 0.2% | 1% | 100% | — | 7 ms |
| M6 speculative decoding | held out | 37 | 5.7% | 26% | 97% | — | 9 ms |
| **All of the above** |  | 182 | 3.4% | 27% | 94% | 3.6% | 8 ms |
| **All held out** |  | 101 | 4.5% | 27% | 92% | 2.5% | 8 ms |
| M6: a separate draft model (not modeled) | held out | 42 | 55.1% | 102% | 12% | — | 62 ms |
<!-- END GENERATED: m8_validation -->

"Held out" points were not used by any fitting stage. The last row is outside the model: a *separate draft
model* inside vLLM costs far more per drafted token than the bytes it reads (M6's open question), and the
model says so by being wrong there. M8 uses an EAGLE head, which the model does cover.

Its largest misses on the points it claims:

<!-- BEGIN GENERATED: m8_worst -->
| Measured in | Model | Server | Workload | Users | Measured tok/s | Predicted | Error |
|---|---|---|---|---|---|---|---|
| M5 KV cache and prefix caching | Qwen3-0.6B | prefix-on | shared_prefix | 8 | 810 | 1,026 | 27% |
| M5 KV cache and prefix caching | Qwen3-0.6B | fp8w-fp8kv | saturation | 512 | 3,492 | 4,421 | 27% |
| M6 speculative decoding | Qwen3-1.7B | bf16-ngram-k6 | spec_mixed | 64 | 1,840 | 1,364 | -26% |
| M4 weight formats | Qwen3-0.6B | int8 | saturation | 512 | 1,852 | 2,231 | 20% |
| M4 weight formats | Qwen3-0.6B | fp8 | saturation | 512 | 1,863 | 2,224 | 19% |
| M4 weight formats | Qwen3-0.6B | gptq | saturation | 512 | 1,836 | 2,189 | 19% |
| M4 weight formats | Qwen3-0.6B | awq | saturation | 512 | 1,842 | 2,189 | 19% |
| M4 weight formats | Qwen3-1.7B | bf16 | decode | 256 | 3,962 | 3,300 | -17% |
<!-- END GENERATED: m8_worst -->

Two patterns: the small model's saturated server (256 sequences of mixed lengths) is slower than predicted,
and uniform batches of 256 are faster. One constant for FlashAttention's batch behaviour cannot serve both.
M8's workloads sit between them.

## 4. Prediction (frozen before any M8 server ran)

What the model says, as speedups over the base server:

**Qwen3-1.7B**

<!-- BEGIN GENERATED: m8_predicted_large -->
| Workload | Base (tokens/s) | `w` | `k` | `p` | `s` | `wk` | `wkp` | `wkps` | `akps` |
|---|---|---|---|---|---|---|---|---|---|
| Latency (1 user, real prompts) | 68 | 1.51× | 1.00× | 1.00× | 1.68× | 1.52× | 1.52× | 2.33× | 3.00× |
| Busy (64 users, real prompts) | 1,899 | 1.33× | 1.15× | 1.00× | 1.26× | 1.61× | 1.61× | 1.89× | 1.68× |
| Capacity (96 users, 4k-token prompts) | 254 | 1.11× | 1.54× | 1.00× | 1.49× | 1.77× | 1.77× | 2.40× | 2.17× |
| Multi-turn (8 users, shared prefixes) | 195 | 1.26× | 1.11× | 1.77× | 1.32× | 1.45× | 2.75× | 3.94× | 4.23× |
| Long (1 user, 32k tokens) | 10 | 1.13× | — | — | — | 1.20× | 1.20× | 1.33× | 1.27× |
<!-- END GENERATED: m8_predicted_large -->

**Qwen3-0.6B**

<!-- BEGIN GENERATED: m8_predicted_small -->
| Workload | Base (tokens/s) | `w` | `wk` | `wkp` | `wkps` |
|---|---|---|---|---|---|
| Latency (1 user, real prompts) | 168 | 1.26× | 1.27× | 1.27× | 1.88× |
| Busy (64 users, real prompts) | 3,550 | 1.14× | 1.59× | 1.59× | 1.97× |
| Capacity (96 users, 4k-token prompts) | 337 | 1.04× | 1.86× | 1.86× | 2.92× |
| Multi-turn (8 users, shared prefixes) | 380 | 1.13× | 1.44× | 2.46× | 3.66× |
| Long (1 user, 32k tokens) | 14 | 1.05× | 1.14× | 1.14× | 1.26× |
<!-- END GENERATED: m8_predicted_small -->

And the ranges I commit to (`benchmarks/predictions/m8.json`), with the reasoning the model does not show:

| Quantity | Prediction | Reasoning |
|---|---|---|
| Full stack on each workload | the model's value, −25% to +10% | The model's errors on old data are mostly within 15%, and unmodeled costs only ever slow things down. |
| INT4 instead of FP8 in the full stack | **faster for one user, slower when busy** | M4's crossover: INT4 reads fewer bytes, FP8 does faster math. |
| FP8 weights × speculation, latency | **0.8–1.0** | They compete: the step shrinks, the drafter's cost does not (M6). |
| FP8 KV × speculation, capacity | **0.7–1.0** | The drafter takes memory the cache would use, and more running sequences leave less idle compute. |
| Prefix caching × speculation, multi-turn | **1.0–1.35** | Complementary: one removes prefill, the other speeds up the decode that remains. |
| FP8 weights × FP8 KV, busy | **0.95–1.15** | Different terms of the step: close to multiplying. |
| Prefix caching where nothing is shared | **0.97–1.03×** | Nothing to cache; the block hashing is cheap. |
| FlashInfer's share of the FP8-KV step, capacity | **10–40%** | M5: part of FP8 KV's gain is its kernel, not its bytes. |
| Tokens per pass on random-token prompts | **1.3–2.8** | Unknown. Text continued from noise could be very predictable (loops) or not at all. |
| Model's median error on M8 | **3–15%** | 4.5% on held-out old data; new combinations and assumed acceptance rates will cost something. |
| Share of M8 points within 15% | **50–90%** | |
| Two runs of the same server | **within 5%** | Closed loops of hundreds of requests are repeatable; the 12-second busy run is the noisiest. |
| Tokens per joule, full stack ÷ base, latency | **1.6–2.6×** | One user leaves the GPU far from its power limit either way; energy per token follows time per token. |
| Base server's power at 64 users | **60–72 W** | Near the L4's 72 W limit. |
| Qwen3-0.6B full stack | below the 1.7B's on latency | Its step has less weight to shrink, and its head is an unknown. |
| Perplexity, FP8 weights + FP8 KV ÷ BF16 | **0.99–1.03** | Each alone was within a percent (M4, M5). |
| Needle recall | **≥ 95%** | FP8 KV kept every needle in M5. |
| GSM8K change | **−6 to +3 points** | FP8 alone was within noise in M4. |

## 5. Result

*Filled in after the measurements.*
