# M1: nanoserve, inference from scratch

M1 builds a small inference engine for Qwen3 in plain PyTorch: **nanoserve** (`src/fastserve/engine/`). It
serves two purposes:
1. It makes every step of inference visible: embeddings, attention, the KV cache, sampling, batching.
2. It's the testbed where every later technique is first built and checked.

It isn't meant to be fast. It's meant to be *correct and readable*, and then to show exactly where the time goes.

## 1. Intuition

Generating text is a loop: **run the whole model to produce one token, append it, repeat.** Two ideas make that
loop affordable:

- **The KV cache.** Every token's keys and values are computed once and kept, so each step only processes the one
  new token. Without it, step *n* would redo the work of all *n* previous tokens.
- **Batching.** One step reads *all* the weights from memory to produce *one* token per sequence. Serving 16
  sequences together reads those same weights once for 16 tokens, which is the same memory traffic for 16× the
  work.

**The paged KV cache** is like virtual memory: instead of reserving the longest possible buffer for every
sequence, the cache hands out small fixed-size blocks as sequences grow and takes them back when they finish.
**Continuous batching** is like a restaurant that seats a new guest the moment a table frees up, instead of
waiting for the whole room to finish.

## 2. The math

For Qwen3 with L layers, hidden size d, H query heads, H_kv KV heads, head size D and vocabulary V:

| Quantity | Formula |
|---|---|
| FLOPs per token (matmuls) | ≈ 2 × parameters |
| Bytes read per decode step (batch B, BF16) | weights + B × context × KV bytes per token |
| KV bytes per token | 2 × L × H_kv × D × 2 |
| Batch-1 decode ceiling | memory bandwidth ÷ bytes per step |
| Arithmetic intensity of a decode matmul at batch B | ≈ B FLOPs/byte |

**The facts for our model** (generated from `config.json` and the M0 bandwidth):

<!-- BEGIN GENERATED: m1_model_facts -->
*Qwen3-0.6B, from config.json; bandwidth from the M0 probe.*

| Quantity | Value |
|---|---|
| Parameters | 596 M |
| Weights in BF16 | 1.192 GB |
| Embedding / LM head share of weights | 26% |
| Bytes read per decode step (batch 1, weights only) | 1.192 GB |
| KV cache per token (BF16) | 112 KiB |
| Batch-1 ceiling = measured read bandwidth ÷ bytes per step | 220 tokens/s |
<!-- END GENERATED: m1_model_facts -->

## 3. What was built

| File | What it does |
|---|---|
| `engine/config.py` | Model dimensions from `config.json`, handling both transformers 4.x and 5.x layouts |
| `engine/rope.py` | Rotary embeddings, matching Hugging Face bit for bit (cos/sin in float32, then cast) |
| `engine/attention.py` | Causal mask *by absolute position*, grouped-query attention, fp32 softmax |
| `engine/kv_cache.py` | `ContiguousKVCache` and `PagedKVCache` (block pool + block tables) behind one interface |
| `engine/model.py` | The forward pass, with Hugging Face's parameter names, so checkpoints load directly |
| `engine/loader.py` | safetensors → model built on the meta device, so there's no double memory |
| `engine/sampler.py` | Greedy, temperature, top-p |
| `engine/generate.py` | Static batching, and `ContinuousBatcher` with "reserve a budget, allocate lazily" admission |

**Correctness checks** (CPU, tiny random-weight models, no downloads):
- logits equal Hugging Face's
- both caches equal a full forward pass
- batched and continuous-batched greedy output equals each prompt run alone

## 4. Prediction (written before running the real model)

The numbers below come from the model facts above plus M0's measurements.

**Memory time.** A decode step reads ~1.19 GB of weights. At the measured 262 GB/s that's 4.6 ms, a ceiling of
~220 tokens/s. But M0 showed that *small* matmuls can't stream at full bandwidth:
- A Qwen3-0.6B layer is seven matrices of 2–6 MiB, streaming at ~110–150 GB/s → about 0.24 ms per layer →
  about 6.8 ms for 28 layers.
- The 311 MB LM head streams faster, ~1.5 ms.

So **GPU memory time is ~8 ms per step** even with perfect execution.

**Kernel count.** Counted by hand from `model.py`, one layer launches roughly:

| Piece | Kernels |
|---|---|
| RMSNorm × 2 (fp32 cast, pow, mean, add, rsqrt, mul, cast back, weight) | ~16 |
| QK-norm (the same, for q and k) | ~16 |
| RoPE for q and k (mul, neg, cat, mul, add) | ~10 |
| q/k/v/o projections, MLP (3 matmuls + SiLU + mul) | ~9 |
| KV-cache write and read, causal mask | ~6 |
| Attention (repeat_kv copies, matmul, scale, mask, softmax, cast, matmul) | ~9 |
| Reshapes that copy, residual adds | ~3 |

That's ~69 per layer × 28 ≈ **1,900 kernels per decode step** (range 1,500–2,500).

**Launch-bound.** M0 measured ~9 µs per eager launch for the simplest op. nanoserve's ops go through more Python,
so figure ~10 µs. 1,900 × 10 µs ≈ 19 ms of CPU work per step, versus ~8 ms of GPU work. The CPU can't queue work
as fast as the GPU finishes it, so:

| Quantity | Prediction | Why |
|---|---|---|
| Batch-1 decode | **35–65 tokens/s** | ~15–30 ms per step, set by launch overhead, not memory |
| GPU busy fraction of a step | **25–60%** | ~8 ms of GPU work inside a ~15–30 ms step |
| Batch 16 vs batch 1 throughput | **12–16×** | The CPU time per step barely changes with batch size, and GPU time stays below it |
| Batch 64 vs batch 1 throughput | **40–64×** | Decode matmuls at M=64 are still memory-bound (64 ≪ ridge 217) |
| Prefill, 512 tokens | **15–35 ms** | ~0.5 TFLOP at ~50 TFLOP/s ≈ 10 ms, but launches still cost ~19 ms |
| Prefill, 2048 tokens | **50–90 ms** | ~2.8 TFLOP (attention grows with length²): compute-bound at last |
| BF16 top-1 agreement with Hugging Face | **99–100%** | Same math and order; BF16 rounding can flip near-ties |
| BF16 max \|logit difference\| | **≤ 0.5** | BF16 has ~3 significant digits, and logits reach ±20 or so |

## 5. Result

*(Measured next; filled in by generated tables.)*

<!-- BEGIN GENERATED: m1_predictions -->
<!-- END GENERATED: m1_predictions -->

## 6. Check your understanding

1. Why does decode at batch 1 read ~1.19 GB per token, even though the embedding table alone is 311 MB?
   <details><summary>Answer</summary>The embedding *lookup* reads one row, but the LM head (the same matrix,
   because the weights are tied) is a full matmul over all 151,936 rows. Every weight of every layer is read once
   per step.</details>
2. Why does nanoserve mask attention by *absolute position* rather than by the row's length?
   <details><summary>Answer</summary>Rows in a batch have different lengths, and padded or not-yet-written cache
   slots hold junk. A key slot j is valid for a query at position p exactly when j ≤ p, so one rule covers
   prefill, decode, padding and paged gathers.</details>
3. What does the paged KV cache buy over the contiguous one, and what does our reference version pay for it?
   <details><summary>Answer</summary>Memory is allocated per block as sequences grow and returned when they
   finish, so no memory is reserved up front and finished sequences free space immediately. Our reference version
   *gathers* the blocks into a contiguous copy each step, which costs extra memory traffic. A real paged-attention
   kernel reads the blocks in place (M7).</details>
4. If a decode step is launch-bound, what happens to its time when the batch grows from 1 to 16? Why?
   <details><summary>Answer</summary>It barely changes: the same kernels are launched regardless of batch size,
   and the GPU work, though bigger, still finishes before the CPU queues the next kernel. So throughput grows
   almost 16×.</details>
5. What two fixes would make nanoserve's batch-1 decode approach the ~220 tokens/s ceiling?
   <details><summary>Answer</summary>Remove launch overhead (CUDA graphs, or fusing small ops like RMSNorm and
   RoPE into single kernels), and stream the small weight matrices closer to full bandwidth (better GEMV kernels,
   or fewer bytes via quantization).</details>

## 7. Further reading

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), 2023.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, 2022.
- Qwen team, *Qwen3 Technical Report*, 2025.
- PyTorch docs: *torch.profiler*; *CUDA semantics: asynchronous execution*.
