# M2: Baselines, stock vLLM under load

M2 measures the **"before" picture**: how a production engine (vLLM, BF16) serves Qwen3-0.6B and Qwen3-1.7B on
one L4, under realistic traffic. Every later optimization is judged against these numbers.

## 1. Intuition

Serving isn't one user running one prompt. It's a **queue**. Requests arrive whenever users send them, wait if the
server is busy, get prefilled, then decode token by token, in a batch with everyone else who's running.
- **At light load** each request is almost alone: latency is as good as it gets, but the GPU is under-used.
- **As load grows** the batch grows, which is nearly free (M1), so throughput climbs while latency barely moves.
- **Past capacity** requests arrive faster than they finish, the queue grows without bound, and time to first
  token explodes.

Where that happens is the **knee**. Service-level objectives (SLOs) turn "fast enough" into numbers, and
**goodput** counts only the requests that meet them.

## 2. The math

| Metric | Definition |
|---|---|
| TTFT | first token arrives − request sent (queueing + prefill + one decode step + network) |
| TPOT | (last token − first token) ÷ (output tokens − 1) |
| ITL | every gap between consecutive tokens (a distribution: p50, p90, p99) |
| Throughput | output tokens ÷ wall time, across all requests |
| Goodput | requests meeting *both* SLOs (TTFT ≤ 500 ms, TPOT ≤ 50 ms) ÷ wall time |
| Cost | ($ per GPU-hour) ÷ (output tokens/s × 3600) × 10⁶ = $ per 1M output tokens |
| Little's law | requests in the system = arrival rate × time in the system |

**A decode step at batch B** reads the weights once, plus every running sequence's KV cache:

```
bytes per step ≈ weights + B × (average context) × KV bytes per token
time per step  ≈ max(bytes ÷ bandwidth, FLOPs ÷ peak) + overheads
throughput     = B ÷ time per step
```

For Qwen3-0.6B the KV cost per token (112 KiB) is as large as for an 8B model, so at high batch the **KV reads**,
not the weights, dominate the bytes.

## 3. Setup

- **Engine:** vLLM 0.30.0 in its own image (`infra/serving.lock`), BF16, default settings except **prefix caching
  off** (it's a technique we add in M5). Its defaults include CUDA graphs, torch.compile, chunked prefill,
  FlashAttention, and up to 256 running sequences.
- **Client:** our async streaming load generator (`serving/client.py`). Prompts are seeded random token ids, and
  output lengths are forced (`ignore_eos`), so every configuration does exactly the same work.
- **Workloads** (`benchmarks/workloads/serving.yaml`):

| Workload | Prompt | Output | Load |
|---|---|---|---|
| chat | ~100 tokens (log-normal) | 256 | one user (closed loop) |
| throughput | ~256 tokens, long tail to 4k | ~190, long tail to 1k | open loop at 1–32 req/s; closed loop 1–256 users |
| long_8k / 16k / 32k | 8k / 16k / 32k | 64 | one user |
| shared_prefix | 2,048 shared + ~64 own | 64 | open loop at 2–24 req/s |

## 4. Prediction (written before the first run)

| Quantity | Prediction | Reasoning |
|---|---|---|
| Chat TPOT p50, 0.6B | **5.5–8 ms** | CUDA graphs remove nanoserve's launch overhead. What's left is GPU time: nanoserve's matmul kernels streamed the 1.19 GB of weights in ~5.4 ms (M1 profile), plus a little for attention, norms and sampling. |
| Chat TTFT p50, 0.6B | **10–30 ms** | A ~100-token prefill is tiny (~0.1 TFLOP ≈ 2 ms). Scheduling, HTTP and the first decode step dominate. |
| vLLM vs nanoserve, batch-1 decode | **6–9×** | ~1000 ÷ 6.5 ms ≈ 150 tok/s, vs nanoserve's 20 tok/s |
| Peak output throughput, 0.6B | **3,000–6,000 tok/s** | At 256 running sequences × ~450 tokens of context, each step reads ~13 GB of KV plus 1.2 GB of weights: ~55 ms per step → ~4,600 tok/s, minus prefill work |
| Knee: highest arrival rate with ≥ 90% of requests within the SLO, 0.6B | **12–25 req/s** | Peak throughput ÷ ~260 tokens per request ≈ 17 req/s |
| Chat TPOT, 1.7B ÷ 0.6B | **2.2–2.9×** | 2.9× the weight bytes, but some fixed per-step costs don't scale |
| Peak throughput, 1.7B ÷ 0.6B | **0.6–0.9×** | At high batch the KV reads dominate, and they're the *same* per token for both models. Only the weights and FLOPs differ. |
| TTFT with a 32k-token prompt, 0.6B | **0.5–1.2 s** | ~29 TFLOP of matmuls + ~4 TFLOP of attention at ~50 TFLOP/s |
| TPOT at 32k context ÷ chat TPOT, 0.6B | **2–4×** | Each step reads 1.19 GB of weights + 3.7 GB of KV (vs ~0.01 GB of KV in chat) |
| Peak request rate, shared-prefix workload, 0.6B, no prefix caching | **12–25 req/s** | Every request re-prefills 2,048 shared tokens (~1.8 TFLOP ≈ 36 ms of compute) |

### Quality baselines (predicted before the quality run)

BF16 is the reference: these numbers aren't "good" or "bad" on their own. They're the bar every later
optimization must not fall below. The ranges are wide because they come from what's typical for models of this
size, not from measurements.

| Quantity | Prediction |
|---|---|
| WikiText-2 perplexity (2,048-token windows) | 0.6B: **12–30**; 1.7B: **9–20** |
| GSM8K 5-shot | 0.6B: **25–55%**; 1.7B: **50–75%** |
| MMLU 5-shot (20 questions per subject) | 0.6B: **35–55%**; 1.7B: **50–65%** |
| HumanEval pass@1 | 0.6B: **10–40%**; 1.7B: **25–55%** |
| Needle-in-a-haystack pass rate, 1k–32k tokens | 0.6B: **70–100%**; 1.7B: **85–100%** |

## 5. Result

*(Filled in by generated tables after the run.)*

<!-- BEGIN GENERATED: m2_predictions -->
<!-- END GENERATED: m2_predictions -->

## 6. Check your understanding

1. Why can throughput go up while TPOT gets worse?
   <details><summary>Answer</summary>A bigger batch makes each step slower (more KV to read, more compute), but
   every step now produces B tokens. Throughput is B ÷ step time and grows as long as B grows faster than the
   step time.</details>
2. What is goodput, and why is it more honest than throughput?
   <details><summary>Answer</summary>Goodput counts only requests that met the latency SLO. Past the knee,
   throughput keeps looking fine while TTFT explodes for everyone. Goodput falls, showing the service is really
   failing its users.</details>
3. Why can a closed-loop benchmark make an overloaded server look healthy?
   <details><summary>Answer</summary>In a closed loop, users wait for their answer before sending again, so a slow
   server automatically receives fewer requests. The queue never runs away. Real public traffic doesn't slow down
   when the server does, and that's what open-loop (Poisson) load models.</details>
4. Why might a 3× bigger model lose much less than 3× peak throughput?
   <details><summary>Answer</summary>At high batch, each step's memory traffic is dominated by the KV cache, which
   depends on layers × KV heads × head size. That's identical for Qwen3-0.6B and 1.7B. Only the smaller part (the
   weights) grows.</details>
5. Why is prefix caching off in the baseline, even though vLLM turns it on by default?
   <details><summary>Answer</summary>To attribute gains honestly. Prefix caching is one of the techniques we
   measure (M5). Leaving it on in the baseline would silently fold its gain into "stock vLLM".</details>

## 7. Further reading

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), 2023.
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (chunked prefill), 2024.
- Zhong et al., *DistServe* (goodput, prefill/decode interference), 2024.
- The vLLM documentation: *Optimization and Tuning*; *Benchmarking*.
