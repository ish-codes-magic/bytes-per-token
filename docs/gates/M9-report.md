# Gate Report: M9, presentation

> **Awaiting the owner's decision.** Not tagged. This is the last milestone: the section "What is left for
> you" lists what only the owner can do.

## What was built

- **The README, finished** ([README.md](../../README.md)): a one-line claim, the final waterfall with a
  computed caption, the findings, a "where to look" table by how much time the reader has, a results section
  per milestone, what was built with links, reproduce instructions, hardware and versions, limitations, and
  what did not work.
- **The results dashboard** ([`site/`](../../site/)): a static page with interactive versions of the key
  figures (waterfall per model and workload, every server on a workload, interactions, the host's chain,
  predicted against measured), every needle grid measured in the project, the token-acceptance highlighter,
  and a **calculator that runs the serving model in the browser**, with a recommendation map.
  - Its data is one generated file ([`site/data/dashboard.json`](../../site/data/dashboard.json), from
    [`report/site.py`](../../src/fastserve/report/site.py) and
    [`scripts/build_site.py`](../../scripts/build_site.py)).
  - Its model ([`site/model.js`](../../site/model.js)) is a port of `perfmodel/serving.py`, checked against
    the Python model's own outputs by the page on load and by the tests.
- **Reproduction** ([ADR 002](../decisions/002-reproduce-without-a-gpu.md)):
  - `make reproduce-quick`: every figure of M0–M8 redrawn from the committed raw records without a GPU, each
    caption required to equal the committed one; the docs' tables and the dashboard's data required to be
    current. CI runs it on every push ([`viz/render.py`](../../src/fastserve/viz/render.py),
    [`scripts/make_figures.py`](../../scripts/make_figures.py)).
  - [`scripts/reproduce_all.sh`](../../scripts/reproduce_all.sh): the full reproduction, written and not run.
  - `make open-question` and [docs/open-questions.md](../open-questions.md): M8's open question as one
    command, with [`scripts/compare_profiles.py`](../../scripts/compare_profiles.py) to read the result.
- **Writing:** the [blog draft](../blog.md), the [glossary](../glossary.md), the
  [interview prep](../interview-prep.md) (a draft to rewrite), a [map of the docs](../README.md), the
  [M9 learning doc](../learning/M9-presentation.md) with a ten-minute talk outline.
- **Summary tables generated from the records** ([`report/summary.py`](../../src/fastserve/report/summary.py),
  `scripts/render_docs.py`): the numbers to know, the versions, and a scoreboard of every prediction in the
  project.
- **Model cards:** the eight published checkpoints' cards on the Hub were fetched and are identical to the
  committed copies in `results/model_cards/`. The project published no drafter of its own; the EAGLE-3
  heads it used are other people's.
- **A workflow that publishes the dashboard to GitHub Pages**, started by hand
  ([`pages.yml`](../../.github/workflows/pages.yml)). Pages is not switched on.

## Key results

M9 measured nothing new. Its results are what is now checked:

<!-- BEGIN GENERATED: m9_reproduction -->
| What is checked | How many | By |
|---|---|---|
| Figures redrawn from the raw records, caption equal to the committed one | 52 | `make reproduce-quick`, in CI on every push |
| Generated tables and captions placed in the README and docs | 325 | `scripts/render_docs.py`; CI fails if the committed docs differ |
| Inputs on which the browser's model must equal the Python model | 192 | `site/tests/parity.mjs`, and the page itself on load |
| Frozen predictions the calculator reproduces from a workload preset | 115 | `site/tests/logic.mjs` |
| Recommendation-map cells equal to the ones Python computes | 126 | `site/tests/logic.mjs` |
<!-- END GENERATED: m9_reproduction -->

The project in numbers:

<!-- BEGIN GENERATED: key_numbers -->
| Quantity | Value | Measured in |
|---|---|---|
| GPU memory bandwidth, read | 262 GB/s | M0 |
| Peak matmul rate, BF16 · FP8 | 57 · 118 TFLOP/s | M0 |
| Ridge point, BF16 | 217 FLOPs per byte | M0 |
| Qwen3-0.6B: weights in BF16 | 1.19 GB (596 M parameters) | M1 |
| Qwen3-0.6B: KV cache per token, BF16 | 112 KiB | M1 |
| Qwen3-0.6B: one-user ceiling, bandwidth ÷ weight bytes | 220 tokens/s | M1 |
| Qwen3-0.6B: stock vLLM, one user · 64 users | 168 · 3,495 tokens/s | M8 |
| Qwen3-1.7B: weights in BF16 | 3.44 GB (1,721 M parameters) | M1 |
| Qwen3-1.7B: KV cache per token, BF16 | 112 KiB | M1 |
| Qwen3-1.7B: one-user ceiling, bandwidth ÷ weight bytes | 76 tokens/s | M1 |
| Qwen3-1.7B: stock vLLM, one user · 64 users | 67 · 1,955 tokens/s | M8 |
| Qwen3-1.7B: best measured stack, latency | `wps`, 2.25× stock | M8 |
| Qwen3-1.7B: best measured stack, capacity | `wkps`, 2.28× stock | M8 |
| Qwen3-1.7B: the full stack, one user | `wkps`, 1.13× stock | M8 |
| Host time per pass on piecewise CUDA graphs | about 25 ms | M8 |
| Serving model, frozen: median error · without speculation | 6.4% · 3.5% | M8 |
<!-- END GENERATED: key_numbers -->

<!-- BEGIN GENERATED: m8_findings -->
- **The best measured stack, Qwen3-1.7B on one L4, against stock BF16 vLLM:** latency `wps` 2.25×, busy `wps` 1.46×, capacity `wkps` 2.28×, multi-turn `wps` 2.76×, long `wkps` 1.22×. With INT4 weights allowed (lower quality, M4): latency `aps` 3.08×, busy `aps` 1.49×, multi-turn `aps` 3.13×.
- **The full stack (`wkps`: FP8 weights, FP8 KV, prefix caching, speculation) is the best stack on 2 of 5 workloads on Qwen3-1.7B and 0 of 5 on Qwen3-0.6B.** At one user it gives 1.13× where `wps` gives 2.25×; on Qwen3-0.6B it is slower than stock (0.48×).
- **Why: one pair collides.** With an FP8 KV cache and speculation together, vLLM 0.30 gives up its full CUDA graph on this GPU, and a step then waits for the host instead of the GPU: 25.8 ms per step against 4.6 with the graph, for the same GPU work (Qwen3-0.6B, FP8 weights, piecewise graphs forced on a control server).
- **Everything else nearly multiplies:** 17 of 24 pair × workload interactions are within 5% of 1. FP8 KV × speculation is 0.77 at one user.
- **Quality of the lossy part of the stack** (FP8 weights + FP8 KV, Qwen3-1.7B, in vLLM): perplexity ×0.996, needle recall 100%, GSM8K -1.3, MMLU -0.4 and HumanEval -6.1 points. Prefix caching does not change outputs; speculation is lossless in distribution (M6).
- **The serving model, frozen before any M8 server ran:** median error 6.4% over 125 predictions (66% within 15%), 3.5% on servers without speculation. Its large misses are the colliding servers: it had no term for the host. With that one constant fitted on M8, the same points come to 5.1% (71% within 15%).
<!-- END GENERATED: m8_findings -->

![Final waterfall](../../results/figures/m8_waterfall.svg)
<!-- BEGIN GENERATED: caption-m8_waterfall -->
*The best measured stack serves Qwen3-1.7B at 1.2–2.8× lower cost per token than stock BF16 vLLM on the same L4 (most on multi-turn, least on long); on 8 of 10 model–workload pairs it is not the full stack, because the last technique added made serving dearer (hatched steps).*
<!-- END GENERATED: caption-m8_waterfall -->

## Predicted vs measured

No prediction was committed before M9's checks ran, so there is nothing to score for this milestone, and none
were written afterwards ([M9 learning doc, section 4](../learning/M9-presentation.md#4-prediction)).

Every range prediction of the project, each written before its measurement:

<!-- BEGIN GENERATED: prediction_scoreboard -->
| Predictions written before measuring | Ranges given | Measured inside the range | Share |
|---|---|---|---|
| M0: the GPU's ceilings | 10 | 4 | 40% |
| M1: nanoserve | 9 | 5 | 56% |
| M2: stock vLLM under load | 10 | 7 | 70% |
| M2: quality baselines | 10 | 10 | 100% |
| M3: quantization from scratch | 24 | 17 | 71% |
| M4: quantized checkpoints in vLLM | 19 | 17 | 89% |
| M5: the KV cache | 31 | 22 | 71% |
| M6: speculative decoding | 28 | 17 | 61% |
| M7: Triton kernels | 29 | 21 | 72% |
| M8: the full stack | 32 | 18 | 56% |
| M8 investigation, round 1 | 8 | 5 | 62% |
| M8 investigation, round 2 | 4 | 3 | 75% |
| M8 investigation: FlashInfer's calls | 5 | 2 | 40% |
| M8 investigation, round 3 | 3 | 0 | 0% |
| M8 investigation, round 4 | 2 | 2 | 100% |
| **All** | **224** | **150** | **67%** |
<!-- END GENERATED: prediction_scoreboard -->

## Surprises and dead ends

- **The dashboard's waterfall was mislabeled and every automated check passed.** The data file's keys were
  sorted alphabetically, and the page paired techniques with ladder steps by position. Found by looking at
  the rendered page. Fixed by deriving each step's technique from its label, and by a test that states the
  order independently.
- **`site/` was ignored by git** through a line of the stock Python `.gitignore` template. The first commit
  of the dashboard would have been empty. Removed.
- **Nothing was stale.** All figures of M0–M8 redrew to their committed captions on the first run in CI.
- **A duplicate test file name** broke collection in CI; renamed.
- **Not built, and why:**
  - *The full reproduction was not run*: the budget (owner's decision, ADR 002).
  - *No Dockerfile*: since ADR 001 the images are defined in `infra/modal_app.py`. Quick reproduce needs no
    image at all.
  - *No separate topic documents `docs/00`–`03`*: the learning docs cover them; `docs/README.md` maps one
    onto the other.
  - *The dashboard is not hosted yet* (below).
  - *The page was checked in one browser*, headless Chrome at desktop width. Its logic is tested in Node;
    its layout on a phone was not looked at.

## What you should now understand

- **Two readers:** what the first screen of a README must hold for one of them, and what the other goes
  looking for.
- **A reproducibility chain:** raw records → tables, figures, captions → pages; which arrows M9 checks, and
  which it does not.
- **Why a caption computed from the data** makes a figure testable.
- **Checking a port against its reference:** identical inputs, then a tight tolerance.
- **A check only finds what it was built to see**: the waterfall bug, and M8's control, are the same lesson.

## Explain it back (answer before we continue)

1. `make reproduce-quick` is green. State precisely what that proves about the project's results, and what it
   leaves unproven.
2. The dashboard's calculator and `perfmodel/serving.py` are two implementations of one model. How is their
   agreement checked, and why is the reference computed from rounded constants?
3. Give the project's result in one sentence for a hiring manager, then in three sentences for an inference
   engineer. What changes between the two, and what must stay the same?
4. Pick the figure you find hardest to explain from the ten-minute outline
   ([M9 learning doc, section 6](../learning/M9-presentation.md#6-the-project-in-ten-minutes)) and explain
   it without the caption.
5. Many of the project's range predictions missed (the scoreboard above). Is that good or bad, and what
   would a score of 100% have told you?

## What is left for you

Things the agent cannot or should not do:

1. **Rewrite the interview answers in your own words** ([docs/interview-prep.md](../interview-prep.md)).
   They are a draft. This is the part of "the human can explain every design choice" that only you can do,
   together with the explain-it-back questions deferred at every gate since M0.
2. **Switch on GitHub Pages** if you want the dashboard hosted: Settings → Pages → Source: "GitHub Actions",
   then run the `pages` workflow (Actions tab, or `gh workflow run pages`). I did not switch it on: it
   publishes a public site under your name.
3. **Run the two GPU jobs kept for later**, when there is budget:
   - `make open-question` (two short profiles, a few minutes of L4 each): why FP8 weights double the host's
     time on piecewise graphs.
   - `bash scripts/reproduce_all.sh` (or one milestone: `bash scripts/reproduce_all.sh m8`): the full
     reproduction.
4. **Read the blog draft as its author.** It is written in the first person, about decisions and mistakes
   that were made in your project. Change what does not sound like you before publishing it anywhere.

Budget, as billed so far:

<!-- BEGIN GENERATED: compute_spend_month -->
   **$22.97** in 2026-10, billed through 2026-10-02 09:00 UTC.
<!-- END GENERATED: compute_spend_month -->

M9 used no GPU. Its tests ran on GitHub Actions. Modal's billing lags by hours, so this figure will still
rise by what M8's last runs cost.

## Definition of done (AGENTS.md §10)

| | Item | Status |
|---|---|---|
| ◐ | All milestones M0–M9 passed their gates and are tagged | M0–M8 tagged (`v0.0`–`v0.8`). M9 awaits this gate. The explain-it-back answers were deferred at every gate, by the owner's choice. |
| ✔ | Every optimized component has a reference implementation and passing tests | Quantizers, KV policies, the rejection sampler and both kernels have plain-PyTorch references and tests against them. The GPU tests were last run in M7; M8 and M9 changed no kernel or engine code. |
| ◐ | Every result has speed + quality + metadata, reproducible from configs | Speed, quality and metadata: yes, for every lossy change. The containers' CPU model is not recorded (the sandbox hides it), which matters for the host-bound servers. Reproducible from configs: written, not re-run. |
| ◐ | README in 60 seconds; docs in depth; dashboard to explore | Built. The dashboard is not hosted until Pages is switched on. |
| ✔ | Performance model validated against measurements | Frozen before M8, then tested; its failure on host-bound servers is documented and a term added. |
| ✔ | Limitations and failed approaches documented | README, docs/06, the gate reports, `JOURNAL.md`. |
| ✔ | Checkpoints and drafters published with model cards | Eight checkpoints with cards. No drafter of our own was trained. |
| ☐ | The human can explain every figure and every line of the kernels | The owner's part: items 1 and 4 above. |

Deviations from AGENTS.md that stand: Qwen3-0.6B and 1.7B on an L4 instead of an 8B model on an H100 (ADR
001); vLLM only, no SGLang baseline; no Docker image; the custom kernels are not inside vLLM (M7 gate);
reproduction as in ADR 002.

## Proposed next steps

The plan ends here. AGENTS.md §6 lists stretch directions, to be taken one at a time and only with your
approval. Ranked by what this project's own results make most worthwhile:

1. **Close the open question** (`make open-question`): cheap, and it finishes M8's explanation.
2. **The collision on Hopper**: three servers on an H100 would show whether the main finding is specific to
   older GPUs, as vLLM's source says. It needs GPU time this project never had.
3. **Adaptive speculation** (§6): M6 and M8 both show speculation helping at low load and costing at high
   load. A controller that switches it by load is the natural follow-up, and the serving model can predict
   where the switch should be.
4. **Mixed-precision bit allocation** (§6) from M3's sensitivity map.
