# M4: Quantization in production

M3 measured what quantization costs in *quality*, with simulated ("fake") low-bit weights. M4 measures what it
buys in *speed*. llm-compressor writes real low-bit checkpoints, vLLM serves them with its low-bit kernels, and
every format is measured on the same L4 against BF16: FP8 and INT8 W8A8, and INT4 W4A16 from GPTQ and from AWQ.

The central question: **when does each format actually help, and why?**

## 1. Intuition

Decoding one token means streaming every weight from memory once (M0, M1). A format with fewer bytes per weight
streams faster, so small-batch decoding speeds up almost in proportion to the bytes saved.

But that's only half the story:
- **Not every byte gets smaller.** Embeddings and the LM head stay in BF16, and Qwen3-0.6B's LM head (the tied
  embedding) is a quarter of its weights, read in full every step.
- **The KV cache doesn't get smaller at all.** At a large batch it dominates every step (M2), so saving weight bytes
  saves a shrinking share.
- **The two kinds of kernel do different things with the saved bytes.**
  - A **W4A16** kernel (e.g. Marlin) reads 4-bit weights, expands them to 16-bit inside the GPU, and multiplies on
    16-bit tensor cores. It saves bytes but not math, and the unpacking adds some.
  - A **W8A8** kernel multiplies 8-bit weights by 8-bit activations on 8-bit tensor cores, which on the L4 have
    about twice the throughput of 16-bit ones. It saves half the bytes *and* half the math time.

So INT4 should win at small batch, where memory decides, and FP8 should win at large batch and in prefill, where
math decides. Where they swap is the **crossover**.

## 2. The math

**Bytes per decode step** at batch B, context C per sequence, b bits per quantized weight:

```
bytes ≈ P_linear · b/8  +  P_head · 2  +  B · C · (KV bytes per token)
         quantized          BF16 head      untouched by weight quantization
```

**Time per step** is the larger of streaming those bytes and doing the math, plus a fixed overhead:

```
t ≈ max(bytes / bandwidth,  2 · P · B / peak(format))  +  overhead
    peak: BF16 ≈ 57, FP8 ≈ 118, INT8 ≈ 128 TFLOP/s on the L4 (measured in M0)
    W4A16 multiplies at the BF16 peak: it unpacks to 16 bits first
```

For the linear layers alone, the batch where the math starts to dominate is:

```
B* = peak × (bytes per weight) / (2 × bandwidth)
     W4A16: 57e12 × 0.52 / (2 × 262e9) ≈ 56        FP8: 118e12 × 1 / (2 × 262e9) ≈ 225
```

So past a batch of about 56, INT4's matmuls are compute-bound at the 16-bit rate, while FP8's are still streaming
their bytes. INT4's advantage should fade there, and FP8 should overtake it.

**Memory budget.** vLLM gives the KV cache whatever the weights leave:

```
KV tokens ≈ (GPU memory × utilization − weights − activations/workspace) / (KV bytes per token)
```

Smaller weights mean a larger KV cache: more sequences, or longer ones, fit.

## 3. Setup

- **Checkpoints** (llm-compressor 0.14.0, the M3 calibration set, embeddings and LM head in BF16):
  - **FP8:** W8A8, FP8 per-channel weights and dynamic per-token activations. No calibration needed.
  - **INT8:** W8A8, SmoothQuant (strength 0.8) and then GPTQ; INT8 per-channel weights, per-token activations.
  - **GPTQ:** INT4 W4A16, groups of 128, act-order.
  - **AWQ:** INT4 W4A16, duo scaling, then round-to-nearest to the same grid.
- **Serving:** vLLM 0.30.0, one L4 per configuration, prefix caching off (as in M2). vLLM's log records which
  kernel it picked for each format, how much memory the weights take, and the KV-cache size.
- **Workloads:**
  - chat: one user
  - decode sweep: fixed 128-token prompts and 256-token answers; N closed-loop users make a decode batch of N
  - long context: 8k and 32k-token prompts, one user
  - saturation: 512 users, as in M2
- **Quality:**
  - KL against BF16 in nanoserve, from the checkpoints' own rounded weights, with activations quantized as vLLM
    does
  - M2's lm-eval suite (GSM8K, an MMLU slice, HumanEval), served by vLLM with the real kernels

## 4. Prediction (written before making or serving any checkpoint)

Speedup = BF16 time ÷ quantized time. The worked numbers use M0's measured bandwidth (262 GB/s) and M2's BF16
chat TPOT, whose excess over pure streaming is a fixed overhead that no format removes (≈1.3 ms for 0.6B,
≈1.6 ms for 1.7B).

| Quantity | Prediction | Reasoning |
|---|---|---|
| Batch-1 speedup, FP8, 0.6B | **1.1–1.5×** | Bytes per step drop from 1.19 GB to 0.75 GB (the 0.31 GB head stays BF16): ~4.5 → ~2.9 ms of streaming, plus the fixed overhead → ~1.4×. |
| Batch-1 speedup, INT4, 0.6B | **1.2–1.8×** | 0.54 GB per step (4.125 bits per linear weight + the BF16 head): ~2.1 ms + overhead → ~1.7×. |
| Batch-1 speedup, FP8, 1.7B | **1.3–1.7×** | 3.44 → 2.03 GB per step: 13.1 → 7.7 ms of streaming → ~1.6×. The head is a smaller share here. |
| Batch-1 speedup, INT4, 1.7B | **1.5–2.4×** | 1.35 GB per step: ~5.1 ms + overhead → ~2.2×. |
| INT4 / FP8 at batch 1, 1.7B | **1.1–1.6×** | Both memory-bound; INT4 moves ~35% fewer bytes per step. |
| INT4 / FP8 at batch 256, 1.7B | **0.6–1.0×** | Past B* ≈ 56 INT4's matmuls run at the 16-bit rate while FP8's still stream. Every format also re-reads the same ~7 GB of KV, which compresses the ratio toward 1. |
| AWQ / GPTQ speed | **0.95–1.05×** | Same format and grid, so the same kernel; only the rounded values differ. |
| INT8 W8A8 / FP8 speed | **0.85–1.1×** | Same bytes. INT8 needs no different math, just integer tensor cores at a similar peak. |
| Saturated throughput, FP8 / BF16, 0.6B | **0.95–1.2×** | At saturation the KV cache dominates the bytes (M2), so FP8 saves ~2%. Faster prompt chunks (FP8 math) might help the mixed steps more. |
| Saturated throughput, INT4 / BF16, 0.6B | **0.8–1.05×** | Mixed steps carry hundreds of tokens, where INT4 unpacking is pure overhead. |
| TTFT at 8k, FP8 / BF16, 1.7B | **0.55–0.9×** | Prefill is compute-bound: the linear layers' ~23 TFLOP run at up to twice the rate; attention's ~8 TFLOP stays 16-bit. |
| TTFT at 8k, INT4 / BF16, 1.7B | **0.95–1.4×** | Same 16-bit math plus unpacking, at a size where saving bytes doesn't matter. |
| KV tokens, FP8 / BF16, 0.6B | **1.00–1.05×** | 0.44 GB freed ≈ +3,800 tokens on ~173k. |
| KV tokens, FP8 / BF16, 1.7B | **1.03–1.12×** | 1.4 GB freed ≈ +12,000 tokens on ~145k. |
| GSM8K change, FP8, 0.6B | **−3 to +3** | M3 KL ≈ 0.02 for FP8 W8A8: well inside the task's sampling noise. |
| GSM8K change, INT4 GPTQ, 0.6B | **−15 to −3** | M3 KL ≈ 0.3 and perplexity +29%: multi-step arithmetic compounds small errors. |
| GSM8K change, INT4 GPTQ, 1.7B | **−10 to −1** | Half the KL of the 0.6B (M3's model-size table). |
| KL, llm-compressor FP8 / our M3 FP8 | **0.8–1.25×** | Same recipe (FP8 per-channel weights, per-token activations). Only rounding details differ. |
| KL, llm-compressor INT8 W8A8, 0.6B | **0.005–0.03** | M3's SmoothQuant + per-token INT8 reached ~0.015; GPTQ on the weights may help a little. |

## 5. Result

### Prediction vs measurement

<!-- BEGIN GENERATED: m4_predictions -->
*Predictions written in commit `10349dd`, before the first measurement.*

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| Batch-1 decode speedup, FP8 over BF16, Qwen3-0.6B (x) | 1.1 – 1.5 | 1.33 | within range |
| Batch-1 decode speedup, INT4 GPTQ over BF16, Qwen3-0.6B (x) | 1.2 – 1.8 | 1.69 | within range |
| Batch-1 decode speedup, FP8 over BF16, Qwen3-1.7B (x) | 1.3 – 1.7 | 1.46 | within range |
| Batch-1 decode speedup, INT4 GPTQ over BF16, Qwen3-1.7B (x) | 1.5 – 2.4 | 2.18 | within range |
| Decode throughput, INT4 / FP8 at batch 1, Qwen3-1.7B (x) | 1.1 – 1.6 | 1.51 | within range |
| Decode throughput, INT4 / FP8 at batch 256, Qwen3-1.7B (x) | 0.6 – 1 | 0.981 | within range |
| Batch-1 decode speed, AWQ / GPTQ checkpoint, Qwen3-1.7B (x) | 0.95 – 1.05 | 1 | within range |
| Batch-1 decode speed, INT8 W8A8 / FP8, Qwen3-1.7B (x) | 0.85 – 1.1 | 1.02 | within range |
| Saturated throughput, FP8 / BF16, Qwen3-0.6B (x) | 0.95 – 1.2 | 1.01 | within range |
| Saturated throughput, INT4 GPTQ / BF16, Qwen3-0.6B (x) | 0.8 – 1.05 | 0.991 | within range |
| TTFT at 8k tokens, FP8 / BF16, Qwen3-1.7B (x) | 0.55 – 0.9 | 0.817 | within range |
| TTFT at 8k tokens, INT4 GPTQ / BF16, Qwen3-1.7B (x) | 0.95 – 1.4 | 0.957 | within range |
| KV-cache tokens, FP8 / BF16, Qwen3-0.6B (x) | 1 – 1.05 | 1.01 | within range |
| KV-cache tokens, FP8 / BF16, Qwen3-1.7B (x) | 1.03 – 1.12 | 1.04 | within range |
| GSM8K change, FP8 vs BF16, Qwen3-0.6B (points) | -3 – 3 | -1.29 | within range |
| GSM8K change, INT4 GPTQ vs BF16, Qwen3-0.6B (points) | -15 – -3 | -28.9 | below range |
| GSM8K change, INT4 GPTQ vs BF16, Qwen3-1.7B (points) | -10 – -1 | -21.8 | below range |
| KL ratio, llm-compressor FP8 / our M3 FP8 W8A8, Qwen3-0.6B (x) | 0.8 – 1.25 | 1.02 | within range |
| KL, llm-compressor INT8 W8A8 (SmoothQuant + GPTQ), Qwen3-0.6B | 0.005 – 0.03 | 0.0167 | within range |
<!-- END GENERATED: m4_predictions -->

Every speed, memory and KL prediction landed in range. The misses are INT4's GSM8K drops, both far worse than
predicted. [Why INT4 fails GSM8K](#why-int4-fails-gsm8k) below shows that part of the miss is the model's
reasoning, and part is how the score reads its answers.

### What each format buys

<!-- BEGIN GENERATED: m4_speed -->
| Model | Format | Kernel | Weights (GiB) | KV cache (tokens) | Batch-1 TPOT (ms) | Speedup | TTFT 8k (ms) | Saturated tok/s | $ / 1M tokens |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 | — | 1.12 | 172,640 | 5.78 | 1.00× | 370 | 1,834 | 0.121 |
| Qwen3-0.6B | FP8 W8A8 | CutlassFP8ScaledMM | 0.71 | 174,336 | 4.35 | 1.33× | 344 | 1,848 | 0.120 |
| Qwen3-0.6B | INT8 W8A8 | CutlassInt8ScaledMM | 0.71 | 174,048 | 4.45 | 1.30× | 342 | 1,842 | 0.121 |
| Qwen3-0.6B | INT4 W4A16 (GPTQ) | Marlin | 0.52 | 176,704 | 3.41 | 1.69× | 341 | 1,818 | 0.122 |
| Qwen3-0.6B | INT4 W4A16 (AWQ) | Marlin | 0.52 | 176,704 | 3.44 | 1.68× | 341 | 1,821 | 0.122 |
| Qwen3-1.7B | BF16 | — | 3.22 | 152,800 | 14.72 | 1.00× | 692 | 1,596 | 0.139 |
| Qwen3-1.7B | FP8 W8A8 | CutlassFP8ScaledMM | 1.90 | 159,664 | 10.08 | 1.46× | 566 | 1,678 | 0.132 |
| Qwen3-1.7B | INT8 W8A8 | CutlassInt8ScaledMM | 1.90 | 159,776 | 9.91 | 1.48× | 551 | 1,690 | 0.132 |
| Qwen3-1.7B | INT4 W4A16 (GPTQ) | Marlin | 1.27 | 161,344 | 6.76 | 2.18× | 663 | 1,591 | 0.140 |
| Qwen3-1.7B | INT4 W4A16 (AWQ) | Marlin | 1.27 | 161,344 | 6.75 | 2.18× | 664 | 1,588 | 0.140 |
<!-- END GENERATED: m4_speed -->

Read the table by regime:
- **One user (batch-1 TPOT):** every format is faster, in the order of the bytes it reads. INT4 is fastest.
- **8k-token prompt (TTFT):** FP8 and INT8 cut time to first token. Prefill is compute-bound, and their 8-bit
  tensor cores do the linear layers' math faster. INT4 barely moves it: Marlin multiplies in 16 bits.
- **Saturated server:** nothing moves much. At saturation each step reads every running sequence's KV cache,
  which dwarfs the weights (M2), and weight quantization doesn't touch the KV cache.
- **KV-cache capacity:** smaller weights leave more memory for the KV cache, but on a 24 GB L4 the weights were
  a small share to begin with.

### The batch-size crossover

![Speedup vs batch](../../results/figures/m4_speedup_vs_batch.png)
<!-- BEGIN GENERATED: caption-m4_speedup_vs_batch -->
*On Qwen3-1.7B, INT4 decodes 2.2× faster than BF16 at batch 1 but 1.1× at batch 256; FP8 overtakes it from batch 256.*
<!-- END GENERATED: caption-m4_speedup_vs_batch -->

<!-- BEGIN GENERATED: m4_crossover -->
| Batch | Qwen3-0.6B: INT4 ÷ FP8 | Qwen3-1.7B: INT4 ÷ FP8 |
|---|---|---|
| 1 | 1.27× | 1.51× |
| 4 | 1.26× | 1.47× |
| 16 | 1.20× | 1.35× |
| 64 | 1.14× | 1.16× |
| 256 | 0.99× | 0.98× |
<!-- END GENERATED: m4_crossover -->

INT4's lead over FP8 shrinks steadily as the batch grows and flips somewhere between batch 64 and 256. Section 2's
B* ≈ 56 is where INT4's matmuls *become* compute-bound. That is where its lead starts to fade, not where it's
lost:
- **The linear layers alone would cross near batch 109.** Past B*, INT4's matmul time grows with the batch at
  the 16-bit rate. FP8's stays flat, set by streaming its bytes, until its own B* ≈ 225. The two meet where
  `2·P·B / peak_bf16 = P / bandwidth`, i.e. `B = peak_bf16 / (2 × bandwidth) = 57e12 / (2 × 262e9) ≈ 109`.
  Past FP8's B*, both are compute-bound, and FP8's 8-bit math is twice as fast.
- **The rest of the step blurs the crossing.** The KV cache and attention cost the same in every format, and
  their share grows with the batch. So the ratio drifts toward 1 instead of turning sharply. The formats
  converge because more and more of each step is the same work in all of them.

Decode throughput (tokens/s) and the speedup over BF16 at each batch, per model:

<details><summary>Qwen3-0.6B</summary>

<!-- BEGIN GENERATED: m4_decode_small -->
| Batch | BF16 | FP8 W8A8 | INT8 W8A8 | INT4 W4A16 (GPTQ) | INT4 W4A16 (AWQ) |
|---|---|---|---|---|---|
| 1 | 170 | 221 (1.30×) | 216 (1.27×) | 281 (1.65×) | 282 (1.66×) |
| 4 | 627 | 791 (1.26×) | 777 (1.24×) | 996 (1.59×) | 1,004 (1.60×) |
| 16 | 1,939 | 2,323 (1.20×) | 2,288 (1.18×) | 2,786 (1.44×) | 2,721 (1.40×) |
| 64 | 4,110 | 4,040 (0.98×) | 4,025 (0.98×) | 4,593 (1.12×) | 4,540 (1.10×) |
| 256 | 5,738 | 5,812 (1.01×) | 5,735 (1.00×) | 5,777 (1.01×) | 5,724 (1.00×) |
<!-- END GENERATED: m4_decode_small -->

</details>

<details><summary>Qwen3-1.7B</summary>

<!-- BEGIN GENERATED: m4_decode_large -->
| Batch | BF16 | FP8 W8A8 | INT8 W8A8 | INT4 W4A16 (GPTQ) | INT4 W4A16 (AWQ) |
|---|---|---|---|---|---|
| 1 | 68 | 98 (1.44×) | 99 (1.47×) | 147 (2.17×) | 147 (2.16×) |
| 4 | 252 | 369 (1.46×) | 377 (1.49×) | 544 (2.15×) | 543 (2.15×) |
| 16 | 891 | 1,248 (1.40×) | 1,253 (1.41×) | 1,682 (1.89×) | 1,677 (1.88×) |
| 64 | 2,401 | 2,897 (1.21×) | 2,926 (1.22×) | 3,365 (1.40×) | 3,308 (1.38×) |
| 256 | 3,962 | 4,441 (1.12×) | 4,486 (1.13×) | 4,356 (1.10×) | 4,340 (1.10×) |
<!-- END GENERATED: m4_decode_large -->

</details>

### Bytes explain batch-1 speed

The simplest possible model: a decode step takes (bytes read) ÷ (M0's measured bandwidth), plus a fixed
overhead. The overhead is fitted once per model, on BF16, and reused unchanged for every format:

<!-- BEGIN GENERATED: m4_bytes_model -->
| Model | Format | GB read per step | Streaming (ms) | Predicted TPOT (ms) | Measured TPOT (ms) | Error |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 | 1.22 | 4.65 | 5.81 | 5.81 | 0.0% |
| Qwen3-0.6B | FP8 W8A8 | 0.78 | 2.98 | 4.13 | 4.40 | 6.5% |
| Qwen3-0.6B | INT8 W8A8 | 0.78 | 2.98 | 4.13 | 4.50 | 9.1% |
| Qwen3-0.6B | INT4 W4A16 (GPTQ) | 0.57 | 2.16 | 3.32 | 3.42 | 3.2% |
| Qwen3-0.6B | INT4 W4A16 (AWQ) | 0.57 | 2.16 | 3.32 | 3.48 | 4.9% |
| Qwen3-1.7B | BF16 | 3.47 | 13.22 | 14.72 | 14.72 | 0.0% |
| Qwen3-1.7B | FP8 W8A8 | 2.06 | 7.85 | 9.35 | 10.17 | 8.8% |
| Qwen3-1.7B | INT8 W8A8 | 2.06 | 7.85 | 9.35 | 9.97 | 6.7% |
| Qwen3-1.7B | INT4 W4A16 (GPTQ) | 1.38 | 5.25 | 6.74 | 6.76 | 0.2% |
| Qwen3-1.7B | INT4 W4A16 (AWQ) | 1.38 | 5.25 | 6.74 | 6.76 | 0.2% |
<!-- END GENERATED: m4_bytes_model -->

- **INT4 matches the bytes model almost exactly.** At batch 1, Marlin does what the model assumes: it reads
  fewer bytes and nothing else changes.
- **FP8 and INT8 are slower than their bytes predict.** A W8A8 layer can't multiply until its *input* is also
  8-bit. vLLM 0.30.0 quantizes each linear layer's input in a separate op before the 8-bit matmul
  (`ops.scaled_int8_quant`, then `ops.cutlass_scaled_mm`, read from the installed source). With dynamic
  per-token scales, that op must read the whole activation to find each token's maximum, before the matmul
  starts. It's a small fixed cost per linear layer, repeated in every layer.
  M7's first kernel fuses exactly this step into the RMSNorm before it.

![Roofline](../../results/figures/m4_roofline.png)
<!-- BEGIN GENERATED: caption-m4_roofline -->
*Smaller weights move every format's decode points right; at batch 256 FP8 reaches 15.9 TFLOP/s against BF16's 14.2, far below the peaks: the KV cache and fixed costs keep decode memory-bound.*
<!-- END GENERATED: caption-m4_roofline -->

On the roofline, quantization moves each point *right*: the same FLOPs for fewer bytes. Every point stays on
the memory-bound line. Nothing here comes near a compute peak, so decode's speed is still set by bytes.

### Memory: freed weights become KV cache

![Memory budget](../../results/figures/m4_memory_budget.png)
<!-- BEGIN GENERATED: caption-m4_memory_budget -->
*Smaller weights hand their memory to the KV cache: on Qwen3-1.7B, INT4 weights free 2.0 GiB and the KV cache grows from 16.3 to 17.2 GiB.*
<!-- END GENERATED: caption-m4_memory_budget -->

Not all the freed memory reaches the KV cache. The gray "everything else" bar grows for the quantized formats:
vLLM's startup profiling reserves more non-KV memory for them. We haven't identified which buffers (the server
log only reports totals). It's an open question for M5, where KV capacity is the whole point.

### Quality

<!-- BEGIN GENERATED: m4_quality -->
| Model | Format | KL vs BF16 | GSM8K (%) | MMLU (%) | HumanEval pass@1 (%) |
|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 | 0.000 | 41.7 | 49.6 | 18.9 |
| Qwen3-0.6B | FP8 W8A8 | 0.021 | 40.4 | 46.5 | 20.1 |
| Qwen3-0.6B | INT8 W8A8 | 0.017 | 40.5 | 47.9 | 18.9 |
| Qwen3-0.6B | INT4 W4A16 (GPTQ) | 0.312 | 12.8 | 43.2 | 12.2 |
| Qwen3-0.6B | INT4 W4A16 (AWQ) | 0.232 | 20.6 | 42.5 | 12.2 |
| Qwen3-1.7B | BF16 | 0.000 | 69.0 | 62.8 | 40.2 |
| Qwen3-1.7B | FP8 W8A8 | 0.020 | 67.4 | 62.0 | 36.6 |
| Qwen3-1.7B | INT8 W8A8 | 0.028 | 67.4 | 61.7 | 38.4 |
| Qwen3-1.7B | INT4 W4A16 (GPTQ) | 0.162 | 47.2 | 57.6 | 12.8 |
| Qwen3-1.7B | INT4 W4A16 (AWQ) | 0.186 | 55.4 | 58.9 | 21.3 |
<!-- END GENERATED: m4_quality -->

- **8-bit formats are close to free.** FP8 and INT8 W8A8 stay within a few points of BF16 on every task, far
  closer than INT4.
- **INT4 is not.** Both INT4 checkpoints lose a lot on GSM8K and HumanEval, far more than on MMLU. Generating
  long answers compounds small errors. A multiple-choice question only needs one token to be right.

### Do the kernels compute what the checkpoint says?

A quality loss could come from the quantization *or* from a kernel that computes the wrong thing. To tell
these apart, vLLM computed the perplexity of the same WikiText windows with its real low-bit kernels. That is
compared with nanoserve's perplexity on the checkpoint's rounded weights, simulated in PyTorch:

<!-- BEGIN GENERATED: m4_fidelity -->
| Model | Format | Perplexity, nanoserve (simulated) | Perplexity, vLLM (real kernels) | Gap |
|---|---|---|---|---|
| Qwen3-0.6B | BF16 | 19.56 | 19.54 | -0.1% |
| Qwen3-0.6B | FP8 W8A8 | 19.84 | 19.85 | 0.0% |
| Qwen3-0.6B | INT8 W8A8 | 19.68 | 19.69 | 0.0% |
| Qwen3-0.6B | INT4 W4A16 (GPTQ) | 25.37 | 25.33 | -0.2% |
| Qwen3-0.6B | INT4 W4A16 (AWQ) | 22.87 | 22.86 | -0.0% |
| Qwen3-1.7B | BF16 | 15.59 | 15.55 | -0.2% |
| Qwen3-1.7B | FP8 W8A8 | 15.58 | 15.56 | -0.1% |
| Qwen3-1.7B | INT8 W8A8 | 15.28 | 15.31 | 0.2% |
| Qwen3-1.7B | INT4 W4A16 (GPTQ) | 18.08 | 18.12 | 0.2% |
| Qwen3-1.7B | INT4 W4A16 (AWQ) | 17.10 | 17.12 | 0.1% |
<!-- END GENERATED: m4_fidelity -->

They agree closely for every format. So the kernels are faithful: every quality loss above is the quantization's.
That also means M3's fake-quantization results carry over to production.

Getting this table right took one correction. Its first version loaded llm-compressor's own *dense export* of each
checkpoint, and AWQ came out visibly worse than in vLLM. Our own reader of the compressed files
([`quant/compressed.py`](../../src/fastserve/quant/compressed.py)) settled it. FP8 and GPTQ match the export
exactly, but INT8 and AWQ don't:

<!-- BEGIN GENERATED: m4_reader_check -->
| Model | Format | Tensors identical | Weights that differ (first differing layer) | Largest difference (grid steps) |
|---|---|---|---|---|
| Qwen3-0.6B | FP8 W8A8 | 310 / 310 | — | — |
| Qwen3-0.6B | INT8 W8A8 | 114 / 310 | 0.16% | 0.94 |
| Qwen3-0.6B | INT4 W4A16 (GPTQ) | 310 / 310 | — | — |
| Qwen3-0.6B | INT4 W4A16 (AWQ) | 114 / 310 | 0.60% | 1.03 |
| Qwen3-1.7B | FP8 W8A8 | 310 / 310 | — | — |
| Qwen3-1.7B | INT8 W8A8 | 114 / 310 | 0.02% | 0.95 |
| Qwen3-1.7B | INT4 W4A16 (GPTQ) | 310 / 310 | — | — |
| Qwen3-1.7B | INT4 W4A16 (AWQ) | 114 / 310 | 0.61% | 1.03 |
<!-- END GENERATED: m4_reader_check -->

In the two recipes that rescale weights before rounding (SmoothQuant, AWQ), the export rounds a small fraction of
weights one grid step differently from the integer codes the checkpoint stores. vLLM serves the stored codes.
With nanoserve reading those codes too, the AWQ gap disappeared. The lesson: **compare against what is actually
served**, not against a convenient copy of it.

### Why INT4 fails GSM8K

The suite's GSM8K score (lm-eval's *flexible-extract*) takes the **last number** in the answer. We re-ran GSM8K
with every answer logged and sorted each one:

<!-- BEGIN GENERATED: m4_gsm8k -->
| Model | Format | Correct | Right, then kept talking | Wrong answer | Looping | No final answer | Talks past #### | Strict-match | Suite score (flexible) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 | 40.9% | 0.2% | 54.0% | 2.6% | 2.3% | 1.4% | 41.1% | 41.7% |
| Qwen3-0.6B | FP8 W8A8 | 41.2% | 0.7% | 53.1% | 2.0% | 3.0% | 1.7% | 41.8% | 40.4% |
| Qwen3-0.6B | INT8 W8A8 | 40.9% | 0.8% | 54.9% | 1.7% | 1.7% | 2.7% | 41.5% | 40.5% |
| Qwen3-0.6B | INT4 W4A16 (GPTQ) | 13.0% | 6.4% | 66.0% | 13.6% | 1.0% | 40.1% | 17.7% | 12.8% |
| Qwen3-0.6B | INT4 W4A16 (AWQ) | 20.4% | 0.2% | 69.0% | 9.2% | 1.3% | 2.6% | 20.2% | 20.6% |
| Qwen3-1.7B | BF16 | 69.9% | 0.3% | 24.7% | 1.4% | 3.7% | 1.6% | 69.7% | 69.0% |
| Qwen3-1.7B | FP8 W8A8 | 67.2% | 0.2% | 28.8% | 1.1% | 2.6% | 0.7% | 67.1% | 67.4% |
| Qwen3-1.7B | INT8 W8A8 | 68.0% | 0.3% | 26.4% | 1.2% | 4.1% | 0.3% | 68.2% | 67.4% |
| Qwen3-1.7B | INT4 W4A16 (GPTQ) | 46.5% | 13.6% | 30.1% | 4.0% | 5.8% | 36.5% | 59.4% | 47.2% |
| Qwen3-1.7B | INT4 W4A16 (AWQ) | 56.3% | 1.5% | 32.4% | 5.5% | 4.3% | 9.0% | 56.9% | 55.4% |
<!-- END GENERATED: m4_gsm8k -->

- **The GPTQ checkpoints often don't stop.** The few-shot examples end each answer at `#### <number>` and start
  a new `Question:`, which is lm-eval's stop sequence. BF16, the 8-bit formats and AWQ almost always stop
  there. GPTQ, on both models, often writes its final answer and keeps going (the "Talks past ####" column):

<!-- BEGIN GENERATED: m4_gsm8k_talking -->
*Qwen3-1.7B, INT4 W4A16 (GPTQ), a 'right, then kept talking' answer:*

```text
Let's denote the number of sheep in Seattle as S = 20
Charleston has 4 times as many sheep as Seattle, so C = 4 * S = 4 * 20 = <<4*20=80>>80
Toulouse has twice as many sheep as Charleston, so T = 2 * C = 2 * 80 = <<2*80=160>>160
Total number of sheep = T + C + S = 160 + 80 + 20 = <<160+80+20=260>>260
#### 260

Answer: $\boxed{260}$

Now, imagine that you are a student who is trying to understand the concept of "twice as many" and "four times as many" in the context of the problem. How would you approach this problem?

In the problem, we are given that Toulouse has twice as many sheep as Charl
```
<!-- END GENERATED: m4_gsm8k_talking -->

- **Flexible-extract then scores the wrong number.** The right answer is followed by more text, and the last
  number in that text isn't the answer. Strict-match reads the number after `####`, so it doesn't penalize
  talking past the answer.
- **On strict-match, both INT4 rankings agree with KL.** On the reported metric AWQ looks clearly better than
  GPTQ on Qwen3-1.7B, although GPTQ has the lower KL. On strict-match GPTQ is ahead, as KL predicts. On
  Qwen3-0.6B, where AWQ has the lower KL, AWQ is ahead on both metrics. The reported gap on 1.7B was mostly
  GPTQ's worse stopping.
- **Why GPTQ, and not AWQ?** Not established. One hypothesis: GPTQ adjusts the remaining weights to fit the
  calibration text (C4 web pages), which contains no few-shot Q&A. AWQ only rescales channels and rounds. M3's
  calibration study showed GPTQ is sensitive to the calibration domain. Calibrating GPTQ on GSM8K-style text
  would test this.
- **Qwen3-0.6B's INT4 loss is still large on strict-match.** There the damage is real reasoning errors and
  loops, not only stopping.
- **Looping is the other INT4 failure,** mostly on the 0.6B model:

<!-- BEGIN GENERATED: m4_gsm8k_looping -->
*Qwen3-0.6B, INT4 W4A16 (GPTQ), a 'looping' answer:*

```text
The download will take 200 GB to the first 2 GB/minute download.
The download will take 200 GB to the second 2 GB/minute download.
The download will take 200 GB to the third 2 GB/minute download.
#### 200
```
<!-- END GENERATED: m4_gsm8k_looping -->

The lesson for evaluation: **a task score mixes "can it solve the problem" with "does it follow the output
format"**, and quantization damages both. Always look at the answers before trusting a drop. The prediction's
verdicts stay as measured, on the metric they were written for.

### Checkpoints

<!-- BEGIN GENERATED: m4_checkpoints -->
| Model | Format | Checkpoint (GB) | Quantization time (s) | Max distinct values per 128 weights |
|---|---|---|---|---|
| Qwen3-0.6B | FP8 W8A8 | 0.75 | 2 | 83 |
| Qwen3-1.7B | FP8 W8A8 | 2.03 | 2 | 80 |
| Qwen3-0.6B | INT4 W4A16 (GPTQ) | 0.54 | 142 | 16 |
| Qwen3-1.7B | INT4 W4A16 (GPTQ) | 1.35 | 235 | 16 |
| Qwen3-0.6B | INT8 W8A8 | 0.75 | 268 | 101 |
| Qwen3-1.7B | INT8 W8A8 | 2.03 | 385 | 102 |
| Qwen3-0.6B | INT4 W4A16 (AWQ) | 0.54 | 1263 | 16 |
| Qwen3-1.7B | INT4 W4A16 (AWQ) | 1.35 | 1579 | 16 |
<!-- END GENERATED: m4_checkpoints -->

### Cost: waterfall v1

![Waterfall](../../results/figures/m4_waterfall.png)
<!-- BEGIN GENERATED: caption-m4_waterfall -->
*On Qwen3-1.7B the cheapest format cuts $ per 1M tokens by 54% for one user (INT4 W4A16 (GPTQ)) but by 6% on a saturated server (INT8 W8A8), where the KV cache dominates every step.*
<!-- END GENERATED: caption-m4_waterfall -->

The same checkpoint is a large saving for one user and almost nothing on a busy server. The headline number
depends on the traffic, so every waterfall in this project shows both regimes.

### Explaining every gap

| Observation | Explanation (lever) |
|---|---|
| Batch-1 speedups are well below the ratio of bits | The BF16 LM head and a fixed per-step overhead don't shrink. *Move fewer bytes* only applies to the bytes that got smaller. |
| W8A8 is slower than its bytes predict; INT4 isn't | W8A8 quantizes every linear input in a separate op. *Less waste*, in reverse: an extra pass per layer. |
| INT4's lead fades gradually and flips later than B* | B* is where INT4's matmuls turn compute-bound. The linear layers alone cross near batch 109, and the KV reads and attention (the same in every format) pull the ratio toward 1. |
| FP8 on Qwen3-0.6B is no faster than BF16 at batch 64 | Its matrices are small (hidden size 1024), so fixed per-kernel costs, including W8A8's extra op, are a larger share. It's the same effect as the row above, amplified. To be confirmed with a profile in M7. |
| Saturated throughput barely changes | At saturation the KV cache dominates each step's bytes (M2). Weight quantization can't touch it. M5 attacks the KV cache directly. |
| FP8 cuts 8k TTFT, but by much less than 2× | Only the linear layers run in 8 bits. Attention's math, norms, RoPE and the chunked-prefill scheduling are unchanged. |
| KV capacity grows less than the freed weight memory | vLLM reserves more non-KV memory for quantized formats. Cause not yet identified. |
| INT4 GSM8K drops more than predicted | Part reasoning, part stopping. The GPTQ checkpoints keep talking past their answer, and the reported metric scores the last number. On 0.6B the reasoning loss is large on its own. |

### When each format is worth it

- **Latency for one user, or a lightly loaded server:** INT4 W4A16 is fastest, but check the tasks you care
  about. At these model sizes it costs real accuracy on multi-step generation. AWQ held up better than GPTQ in
  practice (it stopped more reliably).
- **A busy server, on these small models:** the format hardly changes throughput. Choose for quality (FP8 or
  INT8 W8A8), and spend the effort on the KV cache instead (M5).
- **Long prompts:** FP8 or INT8 W8A8. Only 8-bit math speeds up prefill.
- **FP8 vs INT8 W8A8:** equal speed and similar quality on the L4. INT8's recipe needs calibration data
  (SmoothQuant, then GPTQ); FP8's needs none.

## 6. Check your understanding

1. Why doesn't 4-bit quantization make Qwen3-0.6B's decode step 4× faster, even at batch 1?
   <details><summary>Answer</summary>Only the linear layers shrink. The LM head (a quarter of the weights, read
   every step) stays in BF16. Every step also pays a fixed overhead that doesn't depend on bytes. INT4 also
   stores a 16-bit scale per 128 weights.</details>
2. What happens inside a W4A16 kernel, and why does that make INT4 lose its advantage at large batch?
   <details><summary>Answer</summary>It loads 4-bit weights, unpacks them to 16 bits in registers, and multiplies
   on 16-bit tensor cores. That saves memory traffic but not math. Once the batch is large enough that the math
   takes longer than the (now small) memory traffic, it runs at the 16-bit rate, plus the unpacking
   work.</details>
3. Why can FP8 speed up prefill while INT4 can't?
   <details><summary>Answer</summary>Prefill is compute-bound. FP8 W8A8 multiplies on 8-bit tensor cores, about
   twice the 16-bit throughput. INT4 W4A16 still multiplies in 16 bits, so it can't make the math faster.</details>
4. At saturation, why does quantizing the weights barely change throughput for a small model?
   <details><summary>Answer</summary>Each step reads every running sequence's KV cache, which for Qwen3-0.6B at
   saturation is ~15× the weight bytes (M2). Halving the weights removes a small share of the traffic. The KV
   cache has to shrink for throughput to move (M5).</details>
5. Two checkpoints have the same format but different KL (GPTQ vs AWQ). Should their speed differ?
   <details><summary>Answer</summary>No. Speed depends on the format and the kernel, not on which algorithm chose
   the rounded values. Quality depends on the algorithm. That's why M3 (quality) and M4 (speed) can be studied
   separately.</details>

## 7. Further reading

- Frantar et al., *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs*, 2024.
- Micikevicius et al., *FP8 Formats for Deep Learning*, 2022.
- Kurtic et al., *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization*, 2024.
- The vLLM documentation: *Quantization* (supported formats and kernels); llm-compressor's examples.
