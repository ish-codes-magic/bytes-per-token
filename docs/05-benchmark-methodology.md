# Benchmark methodology

How every speed and quality number in this project is produced, and the pitfalls each choice guards against.
If a result can't be reproduced by following this page, that's a bug.

## Hardware and software

- **One GPU type for every speed number: an NVIDIA L4** (24 GB, Ada), rented per second on Modal. Its measured
  ceilings (M0) are the reference for every "% of peak".
- **Pinned environments.** The research image comes from `uv.lock` (PyTorch 2.14.0). The serving image comes
  from `infra/serving.lock` (vLLM 0.30.0, PyTorch 2.13.0, transformers 5.17.0), on NVIDIA's CUDA 13.0 devel base
  image.
- **Every result carries its metadata** (`fastserve.results`):
  - git commit, and whether the tree had uncommitted changes
  - GPU, driver, CUDA, and the PyTorch / Triton / vLLM / transformers versions
  - the host's CPU count, and Modal's region and cloud provider
  - the full experiment config, the seed and the timestamp
- **Results come only from committed code.** A run that starts with uncommitted changes is flagged as dirty.

## Server configuration (M2 baseline)

- vLLM defaults: CUDA graphs, torch.compile, chunked prefill, FlashAttention 2, up to 256 running sequences.
- **Prefix caching OFF** (`--no-enable-prefix-caching`). vLLM enables it by default, but it's one of the
  techniques we measure (M5). Leaving it on would fold its gain into the baseline.
- The KV-cache size vLLM chooses is parsed from its log and stored with each run.
- **Warm-up:** 8 requests before any measurement, because the first requests pay one-time costs.

## Load generation

- **The client runs in the same container as the server** and talks over localhost, so there's no network
  noise. The price is that they share CPUs (8 vCPUs), which becomes a possible bottleneck at high token rates.
  See pitfalls.
- **Prompts are random token ids**, sent as ids, so no tokenizer is in the loop. Speed depends on lengths, not
  words. The exception is anything that depends on content (prefix reuse, speculative decoding); the shared-prefix
  workload therefore builds its shared prefix explicitly.
- **Output lengths are forced** (`ignore_eos`, temperature 0), so every configuration does exactly the same
  work.
- **The same seeded request list** is used at every load point and for every configuration.
- **Open loop (Poisson arrivals)** measures a public-API-like service, where overload makes queues grow.
  **Closed loop (N users)** measures a fixed set of users, and it hides overload. Both are reported, and the
  knee and goodput use open loop.

## Metrics

Defined in `fastserve/serving/metrics.py`:

| Metric | Definition |
|---|---|
| TTFT | first streamed token − request sent (includes queueing) |
| TPOT | (last token − first token) ÷ (output tokens − 1) |
| ITL | per-token gaps. A streamed chunk carrying k tokens counts as k equal gaps. |
| Throughput | output tokens ÷ (last finish − first send) |
| Goodput | requests with TTFT ≤ 500 ms **and** TPOT ≤ 50 ms, ÷ wall time |
| Knee | the highest open-loop arrival rate at which ≥ 90% of requests meet the SLO |
| Cost | $0.80 per L4-hour ÷ (output tokens/s × 3600) × 10⁶ |

Percentiles (p50/p90/p99) use linear interpolation, as numpy does by default. Averages are never reported alone.

## Quality

- **Perplexity and KL** are teacher-forced, on WikiText-2 (test split), in 2,048-token non-overlapping windows,
  with full-vocabulary distributions from Hugging Face `transformers`.
- **Tasks** run through lm-evaluation-harness on vLLM:
  - GSM8K 5-shot
  - MMLU 5-shot (20 questions per subject)
  - HumanEval 0-shot

  Prompting is base-model style for every configuration. The scores are for comparing configurations, not for
  leaderboards.
- **Needle-in-a-haystack** grid: 1k–32k tokens × 5 depths × 3 secrets, with copyright-free generated filler and
  Qwen3's chat template (thinking off).

## Pitfalls, and how each is handled

| Pitfall | Guard |
|---|---|
| Timing the queueing of GPU work instead of the work | CUDA events for kernel-level timing (M0/M1) |
| Cache hits flattering a benchmark | L2 flushed for memory-bound microbenchmarks; a GPU test fails if a matmul "streams" faster than memory can deliver |
| Averages hiding tails | p50/p90/p99 and CDFs for every latency |
| Closed-loop load hiding overload | Open-loop Poisson load for the knee and goodput |
| A baseline silently including a technique | Prefix caching off; every server argument recorded |
| The client or HTTP layer, not the engine, being the bottleneck | Compare with vLLM's offline engine throughput (no HTTP); check how late the client sends vs its schedule |
| Too-good-to-be-true results | A negative control for exact-match claims (M1); sanity checks against physical limits |
| One-time costs (compilation, cold caches) | Warm-up requests; compile caches on a persistent Volume |
| Results from uncommitted code | Every record carries the commit and a "dirty" flag; runs start from a clean tree |
| Library APIs differing from memory | Checked against the installed version; differences logged in `JOURNAL.md` |
