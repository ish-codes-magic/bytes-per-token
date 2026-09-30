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

*(Written after the runs.)*

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
