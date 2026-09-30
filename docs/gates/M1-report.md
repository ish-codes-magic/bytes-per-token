# Gate Report: M1, nanoserve

> **Gate decision (2026-09-30):** approved by the owner (explain-it-back questions deferred for later study).
> Tagged `v0.1-nanoserve`.

## What was built

- **nanoserve** ([`src/fastserve/engine/`](../../src/fastserve/engine/)), a readable Qwen3 inference engine in
  plain PyTorch:
  - config from `config.json` (transformers 4.x and 5.x layouts)
  - RoPE matching Hugging Face bit for bit
  - grouped-query attention with a causal mask by absolute position
  - contiguous and **paged** KV caches behind one interface
  - greedy/temperature/top-p sampling
  - **static batching** and a **continuous batcher** ("reserve a budget, allocate lazily")
- **Weights** download once into a Modal Volume. The loader builds the model on the meta device, so the weights
  are never copied twice.
- **Experiments** ([`src/fastserve/experiments/m1.py`](../../src/fastserve/experiments/m1.py)):
  - HF parity, with an SDPA negative control
  - decode speed across batch sizes, and prefill speed across prompt lengths
  - a profiler count of kernels and GPU busy time
  - a per-component time split
  - attention maps
  - a real-model continuous-batching run
- **Tests** (CPU, tiny random-weight models, no downloads):
  - logits equal Hugging Face's
  - both KV caches equal a full forward pass
  - batched and continuous-batched greedy output equals solo runs
  - the allocator reuses freed blocks
  - plus a **real-model GPU parity test**
- **Docs:** [M1 learning doc](../learning/M1-nanoserve.md) with predictions committed first (`1042ea5`).

## Key results

<!-- BEGIN GENERATED: m1_speed -->
| Workload | Time per step (ms) | Tokens/s | vs batch 1 |
|---|---|---|---|
| decode, batch 1 | 50.1 | 20 | 1.0× |
| decode, batch 4 | 50.2 | 80 | 4.0× |
| decode, batch 16 | 53.4 | 299 | 15.0× |
| decode, batch 64 | 53.8 | 1,189 | 59.5× |
| decode, batch 256 | 161.1 | 1,589 | 79.6× |
| prefill, 128 tokens | 52.2 | 2,451 | — |
| prefill, 512 tokens | 54.4 | 9,405 | — |
| prefill, 2048 tokens | 358.7 | 5,710 | — |
<!-- END GENERATED: m1_speed -->

<!-- BEGIN GENERATED: m1_profile -->
| Step | GPU kernels | GPU busy (ms) | Step time (ms) | GPU busy | Step time per kernel (µs) | Top kernel types |
|---|---|---|---|---|---|---|
| decode, batch 1 | 1,990 | 8.58 | 50.07 | 17% | 25.2 | matmul 5.4 ms (253), elementwise 1.6 ms (1021), copy / index / cat 1.3 ms (573) |
| prefill, 512 tokens | 2,212 | 24.82 | 54.44 | 46% | 24.6 | matmul 12.1 ms (253), copy / index / cat 5.3 ms (796), elementwise 5.1 ms (1021) |
| prefill, 2048 tokens | 2,184 | 353.86 | 358.65 | 99% | 164.2 | copy / index / cat 139.7 ms (768), elementwise 78.7 ms (1021), matmul 67.3 ms (253) |
<!-- END GENERATED: m1_profile -->

| | |
|---|---|
| ![decode scaling](../../results/figures/m1_decode_scaling.png) | ![anatomy](../../results/figures/m1_anatomy.png) |
| ![roofline](../../results/figures/m1_roofline.png) | ![attention](../../results/figures/m1_attention.png) |

## Predicted vs measured

<!-- BEGIN GENERATED: m1_predictions -->
*Predictions written in commit `1042ea5`, before the first measurement.*

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| BF16 top-1 agreement with Hugging Face (%) | 99 – 100 | 100 | within range |
| BF16 max |logit difference| vs Hugging Face | 0 – 0.5 | 0 | within range |
| GPU kernels per batch-1 decode step | 1,500 – 2,500 | 1,990 | within range |
| Batch-1 decode speed, eager nanoserve (tokens/s) | 35 – 65 | 20 | below range |
| GPU busy fraction of a batch-1 decode step (%) | 25 – 60 | 17.1 | below range |
| Decode throughput, batch 16 / batch 1 (x) | 12 – 16 | 15 | within range |
| Decode throughput, batch 64 / batch 1 (x) | 40 – 64 | 59.5 | within range |
| Prefill of a 512-token prompt (ms) | 15 – 35 | 54.4 | above range |
| Prefill of a 2048-token prompt (ms) | 50 – 90 | 359 | above range |
<!-- END GENERATED: m1_predictions -->

Every gap is explained in the [learning doc, §5](../learning/M1-nanoserve.md#5-result). In short:
- **nanoserve is CPU-bound:** the GPU sits idle most of each step, because every kernel costs far more CPU time
  than the M0 probe's simplest op.
- **Batching is nearly free** until batch 256.
- **Naive attention's length² memory traffic** dominates long prefills.

## Surprises and dead ends

- **Bit-exact parity** with Hugging Face's eager attention in BF16. It looked too good to be true, so we added a
  negative control: against HF's fused SDPA kernel, the logits differ slightly, which shows the comparison can see
  differences.
- **Real ops cost far more CPU time than the M0 launch probe's `x.add_(1)`.** Lesson for future predictions:
  estimate launch overhead from representative ops.
- **A unit test caught my own wrong intuition:** at batch 64 with 128 tokens of context, Qwen3-0.6B's KV cache is
  almost as large as its weights, so decode intensity is well below "≈ batch size".
- **Library behaviors verified against the installed versions:**
  - transformers 5.17 uses `dtype=` and `rope_parameters`
  - huggingface_hub 1.x reports partial downloads as incomplete unless looked up with the same file patterns
- **Deviation from AGENTS.md:** the model is Qwen3-0.6B (ADR 001), so "Llama forward pass" means the Llama
  architecture plus QK-norm.

## What you should now understand

- Every piece of a Qwen3/Llama forward pass: RMSNorm, QK-norm, RoPE, GQA, SwiGLU, tied LM head.
- Why the KV cache exists, how big it is (112 KiB per token here), and when it rivals the weights.
- Contiguous vs paged KV caches, and what continuous batching changes.
- Why an eager PyTorch engine is launch-bound, and why batching is nearly free when it is.
- Why naive attention blows up at long context (motivating FlashAttention).

## Explain it back (for later study)

1. nanoserve's batch-1 decode step launches ~2,000 kernels and the GPU is busy for only a small part of it.
   Why does quantizing the weights (lever 1) *not* speed it up?
2. Batch 64 decodes almost as fast per step as batch 1. Explain why, and what finally changes at batch 256.
3. Why is the LM head, the single biggest matrix, only a tiny share of the step, while RMSNorm is a large one?
4. Where does the 2048-token prefill spend its time, and what would a FlashAttention-style kernel remove?
5. Why do we trust the bit-exact match with Hugging Face, rather than suspecting we compared HF with itself?

## Proposed next steps: M2, baselines

- Stand up **vLLM** on the L4 with Qwen3-0.6B/1.7B (BF16) as the "stock" baseline. Prediction first: CUDA graphs
  should put its batch-1 decode far above nanoserve's.
- Workloads (chat latency, throughput, shared-prefix agent; long-context RAG sized for the L4), and a load
  generator (open and closed loop) writing one JSONL schema.
- Quality harness: KL vs BF16, top-1 agreement, a small lm-eval suite, needle-in-a-haystack.
- Methodology doc: every measurement choice and pitfall.
