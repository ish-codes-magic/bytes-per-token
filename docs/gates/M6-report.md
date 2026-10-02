# Gate Report: M6, speculative decoding

> **Gate decision (2026-10-02):** approved by the owner (explain-it-back questions deferred for later study).
> Tagged `v0.6-speculative`.

## What was built

- **The speculative sampler** ([`spec/rejection_sampler.py`](../../src/fastserve/spec/rejection_sampler.py)),
  with the proof that its output follows the target's distribution in the docstring.
- **Drafters and the draft-verify loop** ([`spec/drafters.py`](../../src/fastserve/spec/drafters.py),
  [`spec/generate.py`](../../src/fastserve/spec/generate.py)): a small model, n-gram prompt lookup, and "no
  drafter" (plain decoding as the k = 0 case).
- **One model interface for speculation** ([`spec/lm.py`](../../src/fastserve/spec/lm.py)): logits for the
  last few context positions. A nanoserve model with a KV cache (rejected drafts are rolled back by
  overwriting) and a toy bigram table both implement it.
- **The losslessness test** ([`tests/test_spec_lossless.py`](../../tests/test_spec_lossless.py), in CI):
  speculative samples against exact distributions, on a toy model and on tiny Qwen3 models. A negative
  control, a drafter whose tokens are always accepted, must fail the same test.
- **An exact replay** ([`spec/simulate.py`](../../src/fastserve/spec/simulate.py)): under greedy decoding,
  one drafter pass over the target's output determines the speculative run for every draft length. A test
  checks it against the real loop.
- **Real prompts** ([`quality/prompts.py`](../../src/fastserve/quality/prompts.py)): chat, code, math and
  summarization. The load generator can now send them, stop where the model stops, and checksum each output.
- **The campaign** ([config](../../benchmarks/configs/m6_spec.yaml)):
  - agreement and replay per task, for three drafters and three target formats
  - the real loop against plain decoding, in BF16 and float32
  - losslessness at temperature 1 on the real models, in BF16 and float32
  - twelve vLLM servers: a draft model at k = 1, 3, 5, an INT4 draft model, n-gram lookup, the public EAGLE-3
    head, and the same drafter against FP8 and INT4 targets
- **Docs:** the [M6 learning doc](../learning/M6-speculative-decoding.md). Its predictions were committed in
  `995a42e`, before any measurement.

## Key results

<!-- BEGIN GENERATED: m6_one_user -->
| Method | Chat | Code | Math | Summarization | Mean | Draft tokens accepted | Tokens per drafted pass | Outputs identical to no speculation |
|---|---|---|---|---|---|---|---|---|
| No speculation | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | — | — | — |
| Qwen3-0.6B drafter, k = 1 | 1.13× | 1.30× | 1.29× | 1.13× | 1.21× | 82% | 1.82 | 44% |
| Qwen3-0.6B drafter, k = 3 | 1.05× | 1.48× | 1.37× | 1.08× | 1.24× | 67% | 3.01 | 44% |
| Qwen3-0.6B drafter, k = 5 | 0.93× | 1.54× | 1.42× | 0.93× | 1.20× | 56% | 3.80 | 41% |
| INT4 Qwen3-0.6B drafter, k = 3 | 0.90× | 1.32× | 1.24× | 0.98× | 1.11× | 63% | 2.90 | 44% |
| EAGLE-3 head, k = 3 | 1.46× | 2.10× | 1.83× | 1.43× | 1.70× | 38% | 2.13 | 41% |
| N-gram lookup, k = 3 | 0.91× | 1.70× | 1.07× | 1.09× | 1.19× | 45% | 2.35 | 56% |
| N-gram lookup, k = 6 | 0.86× | 1.74× | 1.04× | 1.05× | 1.17× | 29% | 2.75 | 44% |
<!-- END GENERATED: m6_one_user -->

<!-- BEGIN GENERATED: m6_batch -->
| Method | KV cache (tokens) | 1 user | 4 users | 16 users | 64 users |
|---|---|---|---|---|---|
| No speculation | 152,800 | 1.00× | 245 tok/s | 820 tok/s | 1,991 tok/s |
| Qwen3-0.6B drafter, k = 1 | 66,096 | 1.21× | 1.22× | 1.06× | 0.88× |
| Qwen3-0.6B drafter, k = 3 | 63,728 | 1.24× | 1.23× | 1.05× | 0.81× |
| Qwen3-0.6B drafter, k = 5 | 63,776 | 1.20× | 1.10× | 0.95× | 0.69× |
| INT4 Qwen3-0.6B drafter, k = 3 | 68,448 | 1.11× | 1.08× | 1.08× | 0.80× |
| EAGLE-3 head, k = 3 | 136,512 | 1.70× | 1.70× | 1.50× | 1.21× |
| N-gram lookup, k = 3 | 138,432 | 1.19× | 1.14× | 1.02× | 0.97× |
| N-gram lookup, k = 6 | 138,272 | 1.17× | 1.11× | 0.97× | 0.92× |
<!-- END GENERATED: m6_batch -->

<!-- BEGIN GENERATED: m6_interaction -->
| Target | Drafter agreement (nanoserve) | Tokens per pass, k = 3 (replay) | Tokens/s without speculation (vLLM) | Draft tokens accepted (vLLM) | Speedup with the drafter, k = 3 |
|---|---|---|---|---|---|
| BF16 | 83.1% | 2.99 | 68 | 67% | 1.24× |
| FP8 W8A8 | 83.1% | 3.01 | 97 | 67% | 0.79× |
| INT4 W4A16 (AWQ) | 82.0% | 2.96 | 145 | 66% | 0.50× |
<!-- END GENERATED: m6_interaction -->

| | |
|---|---|
| ![highlighter](../../results/figures/m6_highlight.png) | ![speedup vs users](../../results/figures/m6_speedup_vs_users.png) |
| ![theory](../../results/figures/m6_theory.png) | ![speedup surface](../../results/figures/m6_speedup_surface.png) |
| ![accepted lengths](../../results/figures/m6_accepted_lengths.png) | ![waterfall](../../results/figures/m6_waterfall.png) |

## Predicted vs measured

<!-- BEGIN GENERATED: m6_predictions -->
*Predictions written in commit `995a42e`, before the first measurement.*

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| Per-token agreement, drafter vs target, all tasks (nanoserve) | 0.55 – 0.8 | 0.831 | above range |
| Agreement on code minus agreement on chat (points of probability) | 0.03 – 0.3 | 0.227 | within range |
| P(agree | previous agreed) − P(agree | previous missed) | 0.05 – 0.3 | 0.135 | within range |
| Tokens per target pass at k=3, all tasks (replay) | 2 – 2.9 | 2.99 | above range |
| Tokens per pass at k=3 ÷ the independent-acceptance formula at the same agreement (x) | 0.9 – 1.15 | 0.965 | within range |
| Tokens per target pass, n-gram drafter at k=6, summarization (replay) | 1.15 – 1.9 | 1.3 | within range |
| N-gram tokens per pass at k=6, code ÷ chat (x) | 1.1 – 2.5 | 1.9 | within range |
| Agreement of the INT4 (AWQ) drafter ÷ the BF16 drafter (x) | 0.85 – 0.98 | 0.967 | within range |
| Agreement with an FP8 target ÷ with the BF16 target (x) | 0.97 – 1.01 | 1 | within range |
| Agreement with an INT4 (AWQ) target ÷ with the BF16 target (x) | 0.88 – 0.99 | 0.986 | within range |
| Target pass over 4 new tokens ÷ over 1, one sequence (nanoserve) (x) | 1 – 1.3 | 1.05 | within range |
| Share of real-loop greedy outputs identical to plain greedy (nanoserve) | 0.75 – 1 | 0.688 | below range |
| Largest chi-square ÷ its 5σ limit over the four prompts (temperature 1) | 0.3 – 1 | 0.586 | within range |
| Largest TV(speculative, plain) ÷ TV(plain, plain) over prompts and prefixes (x) | 0.7 – 1.3 | 25.2 | above range |
| vLLM's acceptance rate at k=3, one user, minus the replay's (points of probability) | -0.06 – 0.06 | 0.00224 | within range |
| Speedup, drafter k=1, one user, mean over tasks (x) | 0.95 – 1.3 | 1.21 | within range |
| Speedup, drafter k=3, one user, mean over tasks (x) | 0.85 – 1.35 | 1.24 | within range |
| Speedup, drafter k=5, one user, mean over tasks (x) | 0.7 – 1.1 | 1.2 | above range |
| Throughput with the INT4 drafter ÷ with the BF16 drafter, k=3, one user (x) | 1.05 – 1.35 | 0.892 | below range |
| Speedup, n-gram k=6, one user, summarization (x) | 1.1 – 1.6 | 1.05 | below range |
| Speedup, n-gram k=6, one user, chat (x) | 0.95 – 1.1 | 0.862 | below range |
| Speedup, EAGLE-3 head k=3, one user, mean over tasks (x) | 1.4 – 2.3 | 1.7 | within range |
| Speedup, drafter k=3, 16 users, mixed tasks (x) | 0.6 – 1 | 1.05 | above range |
| Speedup, drafter k=3, 64 users, mixed tasks (x) | 0.4 – 0.8 | 0.814 | above range |
| Speedup, n-gram k=3, 64 users, mixed tasks (x) | 0.8 – 1.15 | 0.974 | within range |
| Speedup of drafter k=3 on the FP8 target, one user (x) | 0.7 – 1.05 | 0.793 | within range |
| Speedup of drafter k=3 on the INT4 (AWQ) target, one user (x) | 0.5 – 0.85 | 0.501 | within range |
| Share of vLLM outputs with drafter k=3 identical to no speculation, one user | 0.5 – 0.95 | 0.438 | below range |
<!-- END GENERATED: m6_predictions -->

What each drafted token costs inside vLLM, which is what the speedup formula needs:

<!-- BEGIN GENERATED: m6_round_costs -->
| Target | Drafter | Plain decode step (ms) | Target pass with drafting (ms) | Added per drafted token (ms) | Effective c (÷ plain step) | Tokens per pass | Measured speedup |
|---|---|---|---|---|---|---|---|
| BF16 | Qwen3-0.6B drafter, k = 1 | 14.6 | 21.2 | 6.6 | 0.45 | 1.82 | 1.21× |
| BF16 | Qwen3-0.6B drafter, k = 3 | 14.6 | 33.6 | 6.3 | 0.43 | 2.99 | 1.24× |
| BF16 | Qwen3-0.6B drafter, k = 5 | 14.6 | 43.9 | 5.9 | 0.40 | 3.76 | 1.20× |
| BF16 | INT4 Qwen3-0.6B drafter, k = 3 | 14.6 | 36.7 | 7.4 | 0.50 | 2.88 | 1.11× |
| BF16 | EAGLE-3 head, k = 3 | 14.6 | 18.1 | 1.2 | 0.08 | 2.13 | 1.70× |
| INT4 W4A16 (AWQ) | Qwen3-0.6B drafter, k = 3 | 6.7 | 39.0 | 10.8 | 1.60 | 2.97 | 0.50× |
| FP8 W8A8 | Qwen3-0.6B drafter, k = 3 | 10.1 | 36.8 | 8.9 | 0.88 | 2.99 | 0.79× |
<!-- END GENERATED: m6_round_costs -->

## Surprises and dead ends

- **Speculation exposed BF16's rounding.** In BF16 the real loop's greedy output differed from plain
  decoding on some prompts, and at temperature 1 the speculative and plain samplers disagreed on one prompt
  far beyond noise. Both vanish in float32 (the control). The cause: passes of different shapes compute
  measurably different BF16 distributions when top tokens are nearly tied. Each sampler is exact for the
  distribution its own pass computes:

<!-- BEGIN GENERATED: m6_lossless -->
| Arithmetic | Prompt | Samples × tokens | TV between the two samplers' passes | χ², speculative vs its own pass | χ², plain vs its own pass | χ², speculative vs the reference pass | 5σ limit | TV, speculative vs plain samples | TV, plain vs plain samples |
|---|---|---|---|---|---|---|---|---|---|
| BF16 | Chat | 600 × 3 | 0.179 | 0 | 1 | 7 | 12 | 0.168 | 0.020 |
| BF16 | Code | 600 × 3 | 0.000 | 0 | 0 | 0 | 8 | 0.003 | 0.003 |
| BF16 | Math | 600 × 3 | 0.034 | 2 | 2 | 6 | 12 | 0.068 | 0.052 |
| BF16 | Summarization | 600 × 3 | 0.000 | 0 | 0 | 0 | 8 | 0.000 | 0.000 |
| float32 (control) | Chat | 3,000 × 1 | 0.000 | 3 | 2 | 3 | 12 | 0.022 | 0.004 |
| float32 (control) | Code | 600 × 3 | 0.000 | 3 | 1 | 3 | 8 | 0.010 | 0.003 |
| float32 (control) | Math | 600 × 3 | 0.000 | 2 | 0 | 2 | 12 | 0.037 | 0.035 |
| float32 (control) | Summarization | 600 × 3 | 0.000 | 0 | 0 | 0 | 8 | 0.000 | 0.000 |
<!-- END GENERATED: m6_lossless -->

<!-- BEGIN GENERATED: m6_loop -->
| Arithmetic | Prompts | Outputs identical to plain greedy | Rounds identical to the replay | Tokens per pass, real loop | Tokens per pass, replay |
|---|---|---|---|---|---|
| BF16 | 16 | 11 of 16 | 8 of 16 | 3.26 | 3.27 |
| float32 (control) | 16 | 16 of 16 | 16 of 16 | 3.32 | 3.32 |
<!-- END GENERATED: m6_loop -->

- **A measurement that looked like a sampler bug.** Plain sampling "failed" the test against its own exact
  distribution. That can't happen, so the reference was the problem: it came from a differently shaped pass.
- **The cheapest drafter won, not the most accurate.** The EAGLE-3 head accepts far fewer tokens than the
  small model and is much faster, because a drafted token costs it a small fraction of a target step.
- **The INT4 drafter was slower than the BF16 drafter** inside vLLM's speculative loop, though vLLM selected
  the Marlin kernel for it and it is faster standalone. Cause not established.
- **A draft model halves the KV cache.** It keeps its own keys and values for every token.
- **Speculation and quantization compete.** On the INT4 target the same drafter halves the throughput.
- **Dead ends:**
  - A negative control that was too weak (a drafter reporting a uniform q slipped under the limit on tiny
    models). It now forces every draft token to be accepted.
  - The EAGLE-3 head wouldn't load until the whole repo was downloaded: it doesn't ship safetensors.
  - nanoserve can't show the speedup: its drafter step costs as much as its target step (overhead-bound).
    It shows acceptance; vLLM shows speed.
- **Deviations from AGENTS.md:**
  - vLLM only (no SGLang).
  - Target Qwen3-1.7B with Qwen3-0.6B as the draft model (ADR 001).
  - Tree drafting, Medusa and MTP heads: not run.
  - Batch-dependent draft length (`num_speculative_tokens_per_batch_size`): identified in vLLM, not measured.

## What you should now understand

- Why verifying k tokens costs about one step for a single user, and why that stops with a full batch.
- The acceptance rule, the residual distribution, and why the output distribution is unchanged.
- E(α, k), the speedup formula with the drafter's cost c, and why a cheap drafter beats an accurate one.
- Why real acceptances aren't independent, and which way that moves the tokens per pass.
- Why speculation and weight quantization don't multiply.
- What "lossless" means in exact arithmetic, and what BF16 does to it.

## Explain it back (for later study)

1. The EAGLE-3 head had the lowest acceptance rate and the highest speedup. Explain with the speedup formula
   and the round-cost table.
2. Greedy speculative outputs matched plain decoding on every prompt in float32 but not in BF16. What
   differs between a verification pass and a decode step, and why does only BF16 notice?
3. With 64 users the draft model made the server slower. Where did the "free" verification go?
4. The same drafter gave a speedup on the BF16 target and halved the throughput on the INT4 target, with the
   same acceptance rate. Why?
5. What would the negative control in the losslessness test produce if the rejection step were removed from
   the sampler, and why did the first version of that control fail to prove anything?

## Proposed next steps: M7, custom Triton kernels

- **Profile first.** The candidates M4–M6 measured, each with a number attached:
  - W8A8's separate activation-quantization pass (M4)
  - FlashAttention against FlashInfer on mixed prefill/decode batches (M5)
  - the INT4 drafter's step inside the speculative loop (M6)
- **Kernel 1:** fused RMSNorm + FP8 activation quantization.
- **Kernel 2:** decode attention over a quantized KV cache (INT4 KIVI from M5), dequantizing in the kernel.
- **For every kernel:** a PyTorch reference, tests over many shapes, autotuning, and a roofline placement
  against M0's ceilings.
