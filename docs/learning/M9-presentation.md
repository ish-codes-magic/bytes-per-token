# M9: Making the work legible

M0–M8 produced results. M9 produces no new measurement. Its job is to make the results reachable by two
readers who will never run anything: one with a minute, one with an hour, and to make every number they see
checkable.

## 1. Intuition

**Two readers, one repository.**

- The **60-second reader** (a hiring manager) reads the first screen of the README. They need one claim, one
  figure, a few findings, and a reason to believe them.
- The **60-minute reader** (a senior engineer) reads the code, the tests and the methodology. They are
  looking for the place where you fooled yourself. They trust what they could check.

**The inverted pyramid.** Journalists put the conclusion first and the detail after, so a reader can stop
anywhere and still leave with the most important thing they had time for. The README's first screen, the
blog, the results analysis and the raw records are four depths of the same story.

**A figure's caption is its conclusion.** If a figure needs a paragraph to decode, it is a table in
disguise. Every figure in this project carries a one-line caption computed from the data: "the best stack is
X at Y times". When the data changes, the sentence changes, and if it does not, a test fails.

**Why generated numbers matter more than they seem.** A hand-typed number is a claim about the past. Nobody
can tell whether it is still true, including its author. A number produced by a script from a raw file is a
claim that can be re-derived by anyone, and one that breaks loudly when the code or the data move.

**Honesty reads as competence.** The most convincing parts of this repository are the predictions that
missed, the explanation that was wrong, and the list of what did not work. A reader who finds the weak spots
labeled stops looking for hidden ones.

## 2. The math: a chain you can check

The project's output is a chain:

    GPU  →  results/raw/*.jsonl  →  tables, figures, captions  →  README, docs, dashboard

M9 makes the second and third arrows *checkable without a GPU*.

**Check 1: redraw and compare.** Every figure function returns its caption as a string computed from the
records. `make reproduce-quick` redraws all of them from `results/raw/` and requires each redrawn caption to
equal the committed one. Pixels differ between plotting versions; a sentence with the numbers in it does not.
Equal captions mean the committed figure still shows what the records say.

**Check 2: render and diff.** The docs hold generated blocks between markers. Rendering them again must
change nothing (`git diff --exit-code`).

**Check 3: a port must equal its reference.** The dashboard's calculator is the serving model rewritten in
JavaScript. Two implementations of one formula agree only if someone checks. The data file therefore ships
inputs together with the Python model's outputs, and the JavaScript recomputes them.

Two details make that comparison meaningful:

- **Round the inputs first, then compute the reference.** The data file stores constants rounded to 7
  significant digits. If Python computed its answers from the unrounded constants, the two would differ in
  the 7th digit and the tolerance would have to be loose enough to hide real bugs. Computing the reference
  from the *same rounded inputs* lets the tolerance be 10⁻⁹.
- **Why not exactly equal?** Both use IEEE doubles, but `log2` and the order of additions can differ in the
  last bit between a browser and Python. For a sum of n terms the relative error grows like n × 2⁻⁵³, far
  below 10⁻⁹ here.

**Check 4: the page must reproduce what was promised.** Choosing a measured workload in the calculator must
give the prediction that was frozen before M8, and the recommendation map drawn by the page must equal the
one Python draws.

## 3. Worked example: a bug the checks did not catch, and then did

The dashboard's first version built its waterfall from two lists in the data file: the ladder's labels
(`base, w, wk, wkp, wkps`) and the techniques' names. The data file was written with its keys sorted
alphabetically, so the techniques came out as `k, p, s, w`. The page paired them by position. Every bar had
the right height and the wrong name: the step labeled "FP8 KV cache" was FP8 weights.

- The parity check passed: the model was right.
- The data-is-current check passed: the data was right.
- I found it by **looking at the rendered page** next to the figure from M8.

The fix has two parts, and the second matters more:

1. Derive the name from the data, not from position: the technique a step adds is the letter its label gains.
2. Add the test that would have caught it (`site/tests/logic.mjs`): the steps are `w, k, p, s` in that order,
   and each step's ends are the measured servers' values.

The lesson is the same as M8's, in a smaller room: a check only finds what it was built to see. Tests that
compare a thing with itself miss everything they share.

## 4. Prediction

M9 runs no experiment, so there was no measurement to predict, and **no prediction was committed before
M9's checks ran.** Writing ranges now for outcomes I have already seen would be dishonest, so there are none.
What I expected, for the record and without credit: that redrawing every figure in CI would reproduce every
committed caption (it did, on the first run), and that the JavaScript port would match on the first run (it
did). What I did not expect is in section 5.

The project's predictions as a whole, every range written before its measurement:

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

Two things to read in that table. The share is far from 100%, which is what honest ranges look like: a
forecaster who is never wrong is not saying anything. And the worst round is the one that taught the most.

## 5. Result

What is now checked, and how often:

<!-- BEGIN GENERATED: m9_reproduction -->
| What is checked | How many | By |
|---|---|---|
| Figures redrawn from the raw records, caption equal to the committed one | 52 | `make reproduce-quick`, in CI on every push |
| Generated tables and captions placed in the README and docs | 319 | `scripts/render_docs.py`; CI fails if the committed docs differ |
| Inputs on which the browser's model must equal the Python model | 192 | `site/tests/parity.mjs`, and the page itself on load |
| Frozen predictions the calculator reproduces from a workload preset | 115 | `site/tests/logic.mjs` |
| Recommendation-map cells equal to the ones Python computes | 126 | `site/tests/logic.mjs` |
<!-- END GENERATED: m9_reproduction -->

**Surprises**

- **The dashboard's folder was ignored by git.** A line from the stock Python `.gitignore` template
  (`/site`, for a documentation tool this project does not use) silently kept the first commit of the
  dashboard empty. Caught because the commit command reported it; removed.
- **The waterfall's steps were mislabeled** (section 3).
- **A second test file with a name that already existed** broke test collection in CI, where the laptop
  cannot run the suite at all. CI is the only place this project's Python tests run for free, and it caught
  it on the next push.
- **Nothing was stale.** All figures of nine milestones, drawn over four days by code that kept changing,
  redrew to the same captions. That is what generating every number buys.

**What M9 does not show.** That the measurements themselves repeat. The full reproduction is written
(`scripts/reproduce_all.sh`) and was not run: the budget was spent
([ADR 002](../decisions/002-reproduce-without-a-gpu.md)).

## 6. The project in ten minutes

An outline to present from, without notes. One sentence per line is enough; the figures do the rest.

| Minute | Say | Show |
|---|---|---|
| 0–1 | What I did and the one-line result. The best stack is not the full stack. | README headline waterfall |
| 1–2 | Decode is a memory problem: one read of the weights per token. Three levers. | `hw_roofline`, the three-lever table |
| 2–3 | I built the engine and the baseline first. Batching is the second lever. | `m1_anatomy`, `m2_pareto` |
| 3–4 | Move fewer bytes: quantization, and why the format depends on load. | `m4_speedup_vs_batch` |
| 4–5 | The KV cache: capacity, recall, why keys are hard, prefix caching. | `m5_concurrency`, `m5_needle` |
| 5–6 | Speculation: lossless, tested; helps one user, not a busy server. | `m6_speedup_vs_users` |
| 6–7 | Two kernels, and where they lose. | `m7_speedup` |
| 7–9 | Everything together. The collision, the wrong explanation, the control that could not see. | `m8_host_chain`, `m8_interactions` |
| 9–10 | The model: right where the GPU is the limit, blind where the host is. What I would do next. | `m8_predicted` |

If you have two minutes instead of ten: minute 0–1, then minute 7–9.

## 7. Check your understanding

**1. Why compare captions instead of images when checking that a figure is reproducible?**

<details><summary>Answer</summary>

Image bytes change with the plotting library's version, fonts and anti-aliasing, so two correct renders
differ. A caption is a sentence computed from the same data the figure draws, with the key numbers in it. If
the data or the analysis changed, the sentence changes. It tests what matters and ignores what does not.
</details>

**2. The calculator's model is checked against Python's on a list of inputs shipped with the page. Why is
the reference computed from rounded constants?**

<details><summary>Answer</summary>

The browser only ever sees the rounded constants in the data file. If the reference used unrounded ones, the
two would legitimately differ around the 7th digit, and the tolerance would have to be about 10⁻⁶: loose
enough to hide a real mistake in a small term. With identical inputs the only differences left are
last-bit floating-point effects, so the tolerance can be 10⁻⁹.
</details>

**3. The waterfall bug passed every automated check. What kind of test catches it, and why did the existing
ones not?**

<details><summary>Answer</summary>

The existing tests checked that the model and the data were right, and they were. The bug was in how the
page *paired* two correct lists. The test that catches it states an expectation that does not come from the
code under test: the steps must be FP8 weights, FP8 KV, prefix caching, speculation, in that order, with
these servers' values at each end.
</details>

**4. A reader has 60 seconds. What must the first screen of the README contain, and what must it not?**

<details><summary>Answer</summary>

One claim, one figure whose caption states the result, three to five findings, and how the claim is framed
(better configurations of vLLM for a workload, on this GPU). It must not contain setup instructions, the
project's history, or a result without its quality cost. Everything else is one click away.
</details>

**5. `make reproduce-quick` passes. What does that prove, and what does it not?**

<details><summary>Answer</summary>

It proves that every figure, table and dashboard number follows from the committed raw records by the
committed code. It does not prove the records themselves: that running the benchmarks again on a GPU would
give the same numbers. For that there is one repeated server pair in M8 and a script that was not run.
</details>

## 8. Further reading

- Sandve, Nekrutenko, Taylor and Hovig, *Ten Simple Rules for Reproducible Computational Research*, PLoS
  Computational Biology, 2013. Rule 1 is this milestone: for every result, keep track of how it was produced.
- Tufte, *The Visual Display of Quantitative Information*. Data-ink, small multiples, and why a chart should
  answer one question.
- Hoefler and Belli, *Scientific Benchmarking of Parallel Computing Systems*, SC 2015. Twelve rules for
  reporting performance results; most of this project's measurement rules are in it.
- Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, 1991. Why two correct
  implementations differ in the last bit.
