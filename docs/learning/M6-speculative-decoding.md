# M6: Speculative decoding

M4 and M5 moved fewer bytes per token. M6 uses the second lever: **more tokens per byte moved**. A decode step
streams the whole model to produce one token. Speculative decoding makes that one pass check several guessed
tokens at once, and keeps the ones the model itself would have produced.

Two claims are tested here:
1. **It is lossless.** The output has exactly the distribution the target model alone would give. This is
   proved, then tested statistically.
2. **It only pays when there is slack.** A lightly loaded GPU has spare compute during every memory-bound
   step. A busy GPU doesn't, and speculation then costs more than it gives.

## 1. Intuition

**The waiter and the chef.** A chef (the target model) can only confirm one dish at a time, and each
confirmation means a walk to the pantry (streaming the weights). A waiter (the drafter) who knows the menu
guesses the next few orders. The chef checks all the guesses in one walk. Every correct guess saves a walk; a
wrong guess wastes nothing but the waiter's effort, because the chef says what it should have been.

**Why checking k tokens costs about one step.** At batch 1 a decode step is memory-bound (M1): its time is the
time to stream the weights. Feeding k + 1 tokens through that same pass reads the weights once, as before.
The extra math fits in compute the GPU wasn't using.

**Why nothing is lost.** With greedy decoding it's obvious: a guessed token is kept only if it is the token
the target would pick. With sampling it takes a rule (section 2): accept a guess with probability
min(1, p/q), otherwise resample from what the drafter under-represented. The guesses change *when* tokens are
computed, never *which* tokens are likely.

**Drafters**, from most to least expensive:
- **A smaller model** (Qwen3-0.6B for Qwen3-1.7B): good guesses, but each guessed token costs a small-model
  step.
- **A head on the target's own features** (EAGLE): one extra layer reusing the target's last hidden state.
  Much cheaper per guess, and it sees what the target "thinks".
- **N-gram prompt lookup**: no model. If the last few tokens appeared earlier in the context, guess that what
  followed them will follow again. Free, and only right when the output repeats the input.

**Where it stops working.** With many sequences in the batch, the step is no longer waiting on memory: the
compute is in use. Verifying k + 1 tokens per sequence then multiplies the step's work, and the drafter's own
steps come on top.

## 2. The math

**The acceptance rule.** The drafter draws token x from its distribution q. The target's distribution for
that position is p.

```
accept x with probability min(1, p(x) / q(x))
if rejected: draw the replacement from  norm(max(0, p − q))
```

**Why the output follows p.** For any token x:

```
P(output = x) = P(drafted x and accepted) + P(rejected) · P(replacement = x)
              = q(x) · min(1, p(x)/q(x))  +  R · max(0, p(x) − q(x)) / Z
              = min(p(x), q(x))           +  max(0, p(x) − q(x))              (R = Z, see below)
              = p(x)
```

R = P(rejected) = 1 − Σ min(p, q), and Z = Σ max(0, p − q). They are equal: the probability mass where p
exceeds q is the same as the mass where q exceeds p, since both sum to 1.

**Tokens per target pass.** If each draft token is accepted independently with probability α, a round with k
drafted tokens yields

```
E(α, k) = 1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α)        tokens
```

(the "1" is the token the target always contributes: the replacement, or a bonus token after k acceptances).

**Speedup.** Let c = (one drafter step) ÷ (one target step) and v = (a target pass over k + 1 tokens) ÷ (a
pass over 1). A round costs k·c + v target-steps and yields E tokens:

```
speedup = E(α, k) / (k·c + v)
```

- Memory-bound (one user): v ≈ 1, so speedup ≈ E / (k·c + 1). It needs a cheap drafter (small c) and good
  guesses (high α).
- Busy server: v grows toward k + 1 (every verified token is real compute), and the drafter's steps are no
  longer free. Speedup falls below 1.

**The greedy case and the replay.** Under greedy decoding the output is the target's own text y. A drafted
token at position j is accepted iff it equals y[j], and while the drafts match, the drafter has seen exactly
the target's context. So one pass of the drafter over y gives a bit per position, "would the drafter have
guessed y[j]?", and those bits determine the speculative run for *every* k
([`spec/simulate.py`](../../src/fastserve/spec/simulate.py)). M6 measures agreement once and replays it.

## 3. Setup

- **Reference (nanoserve):**
  - the rejection sampler: [`spec/rejection_sampler.py`](../../src/fastserve/spec/rejection_sampler.py)
  - drafters (a model, n-gram lookup): [`spec/drafters.py`](../../src/fastserve/spec/drafters.py)
  - the draft-verify loop: [`spec/generate.py`](../../src/fastserve/spec/generate.py)
  - One interface serves real models (with a KV cache that rolls back rejected drafts by overwriting them)
    and a toy bigram model whose exact output distribution is known.
- **Losslessness test** ([`tests/test_spec_lossless.py`](../../tests/test_spec_lossless.py), in CI):
  speculative samples against the *exact* target distribution on the toy model and on tiny Qwen3 models
  (chi-square). A negative control, a drafter whose tokens are always accepted, must fail the same test.
  On the GPU the same measurement runs on the real models at temperature 1.
- **Models:** target Qwen3-1.7B. Drafters:
  - Qwen3-0.6B (same tokenizer)
  - M4's INT4 (AWQ) Qwen3-0.6B
  - n-gram lookup (match the last 2–4 tokens)
  - in vLLM only: the public EAGLE-3 head `AngelSlim/Qwen3-1.7B_eagle3`
- **Prompts:** real text, 4 tasks ([`quality/prompts.py`](../../src/fastserve/quality/prompts.py)): chat
  (Dolly), code (HumanEval), math (GSM8K), summarization (CNN/DailyMail). Greedy decoding, thinking off, up
  to 192 new tokens.
- **Production (vLLM 0.30.0, L4):** `--speculative-config` with `draft_model` (k = 1, 3, 5), `ngram`
  (k = 3, 6), `eagle3` (k = 3). Loads: one user per task, then mixed tasks at 4, 16 and 64 users. vLLM's own
  counters give the acceptance per draft position. A checksum of each output checks that speculation didn't
  change the text.
- **Interaction experiment:** the same drafter against M4's FP8 and INT4 targets, in nanoserve (agreement) and
  in vLLM (speed).

## 4. Prediction (written before any M6 measurement)

Inputs from earlier milestones (vLLM, one user): a Qwen3-1.7B step takes 14.7 ms, a Qwen3-0.6B step 5.8 ms
(INT4: 3.4 ms). So **c ≈ 0.39** for the BF16 drafter. That is expensive: published setups use c < 0.1.

| Quantity | Prediction | Reasoning |
|---|---|---|
| Per-token agreement, drafter vs target | **0.55–0.80** | Same family and training data; a third of the parameters. |
| Agreement, code − chat | **+0.03 to +0.30** | Code has boilerplate and repeated identifiers; open-ended prose doesn't. |
| Burstiness: P(agree after agree) − P(agree after miss) | **0.05–0.30** | Easy stretches come in runs. |
| Tokens per pass at k = 3 | **2.0–2.9** | E(0.65, 3) = 2.35 if acceptances were independent. |
| Tokens per pass ÷ the independent formula | **0.9–1.15×** | Runs help long drafts and hurt after a miss; roughly a wash at k = 3. |
| N-gram tokens per pass, k = 6, summarization | **1.15–1.9** | Summaries reuse names and phrases, but only in short runs. |
| N-gram, code ÷ chat | **1.1–2.5×** | "Reply with the whole function" makes the model copy the stub. Chat has nothing to copy. |
| INT4 drafter's agreement ÷ BF16 drafter's | **0.85–0.98×** | AWQ INT4 0.6B has KL ≈ 0.23 from BF16 (M4): its guesses are a little worse. |
| Agreement with FP8 target ÷ with BF16 target | **0.97–1.01×** | The FP8 target is nearly the same model (KL ≈ 0.02). |
| Agreement with INT4 target ÷ with BF16 target | **0.88–0.99×** | The INT4 target's own choices are noisier (KL ≈ 0.18), and the drafter guesses the *unquantized* behaviour. |
| Verifying 4 tokens ÷ 1 token, one sequence | **1.0–1.3×** | Memory-bound: the weights are read once either way. |
| Real-loop greedy outputs identical to plain | **0.75–1.0** | Exact in exact arithmetic. In BF16, a k+1-token pass and a 1-token pass round differently, which can flip near-ties. |
| Worst chi-square ÷ its limit (temperature 1) | **0.3–1.0** | A correct sampler stays under the limit. |
| Worst TV(spec, plain) ÷ TV(plain, plain) | **0.7–1.3×** | Indistinguishable from the noise between two plain samples. |
| vLLM acceptance − replay acceptance (k = 3) | **−0.06 to +0.06** | Same models, same prompts, same rule. |
| Speedup, drafter k = 1, one user | **0.95–1.3×** | E ≈ 1.65 over a cost of 0.39 + 1.05 = 1.44 → 1.15×. |
| Speedup, drafter k = 3, one user | **0.85–1.35×** | E ≈ 2.35 over 3 × 0.39 + 1.05 = 2.22 → 1.06×. |
| Speedup, drafter k = 5, one user | **0.7–1.1×** | E ≈ 2.64 over 5 × 0.39 + 1.05 = 3.0 → 0.88×. Long drafts from an expensive drafter lose. |
| INT4 drafter ÷ BF16 drafter, k = 3 | **1.05–1.35×** | c drops from 0.39 to 0.23; agreement drops a little. |
| Speedup, n-gram k = 6, summarization | **1.1–1.6×** | The drafter is free, so the speedup is about its tokens per pass. |
| Speedup, n-gram k = 6, chat | **0.95–1.1×** | Few matches: nothing gained, little lost. |
| Speedup, EAGLE-3 k = 3 | **1.4–2.3×** | One extra layer per guess (c ≈ 0.05–0.1) with acceptance like a small model's. |
| Speedup, drafter k = 3, 16 users | **0.6–1.0×** | Verifying 64 tokens costs like a 64-sequence step (M4: 27 ms vs 18 ms), plus 3 drafter steps. |
| Speedup, drafter k = 3, 64 users | **0.4–0.8×** | Verifying 256 tokens ≈ 65 ms vs 27 ms plain, plus ~47 ms of drafting, for ~2.35 tokens: ≈ 0.56×. |
| Speedup, n-gram k = 3, 64 users | **0.8–1.15×** | It only verifies when it has a match, so little compute is wasted. |
| Speedup of the drafter on the FP8 target | **0.7–1.05×** | The target got faster (10.1 ms), the drafter didn't: c = 0.57. |
| Speedup of the drafter on the INT4 target | **0.5–0.85×** | c = 0.86: drafting three tokens costs more than the passes it saves. |
| vLLM outputs identical to no speculation | **0.5–0.95** | Lossless in distribution; in BF16 a near-tie can flip, and the text diverges from there. |

## 5. Result

*(Written after the runs.)*

## 6. Check your understanding

1. Why does verifying k drafted tokens cost about the same as generating one, for a single user?
   <details><summary>Answer</summary>A decode step at batch 1 is memory-bound: its time is the time to stream
   the weights (and the sequence's KV cache) once. Passing k + 1 tokens through the same step reads those
   bytes once too; the extra arithmetic uses compute that was idle. It stops being true when the batch is
   large enough that the step is compute-bound.</details>
2. Show, informally, that speculative sampling doesn't change the output distribution.
   <details><summary>Answer</summary>A token x comes out either by being drafted and accepted, with
   probability q(x)·min(1, p(x)/q(x)) = min(p(x), q(x)), or as a replacement after a rejection, with
   probability max(0, p(x) − q(x)) (the rejection probability and the normalizer of the replacement
   distribution are the same number, so they cancel). The two add up to p(x).</details>
3. A drafter is right 70% of the time and costs 0.4 target-steps per token. Is k = 5 better than k = 1?
   <details><summary>Answer</summary>k = 1: E = 1.7 tokens for 0.4 + 1 = 1.4 steps → 1.21×. k = 5:
   E = (1 − 0.7⁶)/0.3 ≈ 2.94 tokens for 2 + 1 = 3 steps → 0.98×. The later draft tokens are rarely reached
   (0.7⁵ ≈ 17%) but always paid for. With an expensive drafter, short drafts win.</details>
4. Why does speculative decoding help less, or hurt, when the server is busy?
   <details><summary>Answer</summary>The gain came from idle compute during memory-bound steps. With a large
   batch the compute is busy: verifying k + 1 tokens per sequence multiplies the work of each step, and the
   drafter's steps take real time too. The same GPU time would have produced more tokens by just decoding.
   </details>
5. Why might quantizing the target lower the acceptance rate, and why does it lower the *speedup* even if
   acceptance stays the same?
   <details><summary>Answer</summary>Acceptance: the quantized target's choices deviate from the original's
   at uncertain positions, and the drafter was trained to imitate the original. Speedup: a quantized target
   is faster per step, so the drafter's unchanged cost is a larger fraction of it (c grows). Both techniques
   spend the same slack: the idle time of a memory-bound step.</details>

## 7. Further reading

- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, 2023.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023.
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024, and *EAGLE-3*, 2025.
- Saxena, *Prompt Lookup Decoding*, 2023 (n-gram drafting).
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024.
- The vLLM documentation: *Speculative Decoding*.
