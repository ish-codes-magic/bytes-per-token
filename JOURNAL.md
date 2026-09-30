# Journal

A dated, append-only log of decisions, dead ends and surprises. Dead ends stay here for good.

## 2026-09-29

- **Decision: cloud-only compute, small models, one L4 GPU.** See
  [ADR 001](docs/decisions/001-cloud-only-small-models.md). Two plans were dropped before any code was written:
  - H100 + Llama-3.1-8B: over budget.
  - A 4 GB laptop GPU + Llama-3.2-1B: its models and toolchain didn't fit the laptop's free disk.
- **Decision: Qwen3-0.6B/1.7B instead of SmolLM2.** SmolLM2's hidden sizes (576, 960) aren't multiples of 128, and
  production INT4 kernels need that.
- **Library check (AGENTS.md §2.3):** Modal 1.6.0's `Image.uv_sync` installs the dependencies from `uv.lock` but not
  the project itself. Our source is added with `add_local_dir` plus `PYTHONPATH`.
- **Watch item:** `uv lock` resolved PyTorch 2.14.0, which is built against CUDA 13.0. That needs a recent NVIDIA
  driver on the GPU host. Verify on the first Modal run.
- **Deferred:** the Dockerfile (AGENTS.md M0) moves to M9. Until then the Modal image, built from the same lock
  file, is the reproducible environment.
- **Surprise:** Modal won't start *any* function in an app that contains an L4 function until the account has a
  payment method, even on the free-credit Starter plan. CPU tests were verified through a throwaway CPU-only app:
  38 passed, 3 GPU tests skipped.
- **Surprise (cost risk):** if a container fails *while starting* (for example on an import error), Modal keeps
  restarting it for as long as the local `modal run` is alive. Function timeouts don't cover startup failures.
  **Rule: always run cloud commands under a local time limit** (e.g. `timeout 900 ...`).
- **Windows quirk:** the Modal CLI prints "✓", which the default Windows console encoding can't handle. Run with
  `PYTHONUTF8=1`.

## 2026-09-30

- **Resolved watch item:** PyTorch 2.14.0 (CUDA 13.0) runs on Modal's L4 driver. All 3 GPU tests pass, including a
  quick probe run.
- **Cost check:** after the local `modal run` process is killed, its cloud app keeps running for up to about a
  minute before Modal notices. After any interrupted run, check `modal app list` (and `modal app stop <id>`).
- **Process:** the first full probe was stopped before finishing because the probe code wasn't committed yet, so its
  records would have pointed at a commit that doesn't contain the code. Probe runs now start from a clean tree.
- **Bug (cost one probe run):** the full probe finished on the L4, but its results couldn't be unpickled on the
  laptop. `torch.__version__` is a `TorchVersion` (a `str` subclass), and unpickling it needs PyTorch, which the
  laptop deliberately lacks. Fix: record versions with `str()`, and pass every cloud function's return value through
  a JSON round trip (`results.to_plain`), so only plain types cross the boundary. There's a test for each.
- **First full probe (run `8088e303e4bc`, kept in `results/raw/hw_probe.jsonl`).** Two measurement flaws, found by
  the "too good to be true" rule:
  1. *Cache hits in matmul timing.* BF16 M=1, N=K=4096 reached 0.74 TFLOP/s. That means its 32 MiB weight was
     streaming at ~745 GB/s, almost 3× the measured memory bandwidth (262 GB/s). The weight fits in the 48 MiB L2,
     so repeated iterations never touched memory. At N=K=8192 (128 MiB, too big for L2) the same shape streams at
     ~260 GB/s, i.e. exactly memory speed. Fix: flush L2 before every timed matmul (outside the timed region). Real
     decode reads ~1.2 GB of weights per token and can never live in L2.
  2. *Contaminated idle power.* "Idle" was sampled right after heavy work while clocks were still high (mean 37 W,
     max 63 W). A true idle reading at the start was 18 W. Fix: measure idle first.
- **Finding: the L4 is power-limited.** Sustained BF16 matmuls sit at the 72 W cap, and the SM clock drops to
  990 MHz (maximum 2,040). The datasheet's 121 TFLOP/s assumes the maximum clock. Scaled to 990 MHz that's
  ~59 TFLOP/s, and the biggest matmul measured 57.9. Medium matmuls reach higher (71 TFLOP/s) because they finish
  before the power limiter pulls the clock down. Next run: sample the SM clock during each power workload so the
  table shows this directly.
- **Limitation:** Modal's gVisor sandbox hides the host CPU model (`/proc/cpuinfo` has no "model name"). Recorded as
  "unknown".
- **M0 results (run `aed53c62b464`):** memory bandwidth landed inside the predicted range. BF16 compute fell below
  it, because the L4 holds its SM clock at about half the maximum under sustained tensor math (power cap). Scaled to
  that clock, cuBLAS is within a few percent of the hardware limit. Decode-shaped (M=1) matmuls stream their weights
  well below the best-case bandwidth, and the smaller the weight the worse. M1's decode prediction must use those
  per-matmul rates.
- **Open question:** the SM clock wasn't sampled during FP8 matmuls. Add an FP8 power workload to the probe when
  M4 needs it.
- **Gate M0 approved** (questions deferred by the owner). Tagged `v0.0-foundations`.
- **Library check (AGENTS.md §2.3):** huggingface_hub 1.x reports a partial download as an
  `IncompleteSnapshotError` under `local_files_only=True`, unless the lookup passes the same `allow_patterns` as the
  download. All lookups now go through `engine.loader.model_dir`.
- **transformers 5.17:** `from_pretrained` takes `dtype=` (not `torch_dtype=`), and configs store RoPE settings in
  `rope_parameters`. The real Qwen3-0.6B `config.json` still uses the 4.x layout, so `ModelConfig.from_hf` accepts
  both.
- **M1 results (runs `3b2ec4535995` and `d32c18962945`, which agree closely):**
  - nanoserve matches Hugging Face's eager attention **bit for bit** in BF16. The negative control (HF's fused
    SDPA kernel) differs by up to about 1.3 in the logits, so the comparison can detect differences.
  - Batch-1 decode is ~20 tokens/s: ~2,000 kernels per step at ~25 µs of CPU time each, with the GPU busy
    ~17% of the step. My prediction assumed ~10 µs per kernel, extrapolated from M0's cheapest op (`x.add_`).
    **Lesson:** estimate launch cost from representative ops.
  - Batching is nearly free up to 64. At 256 the GPU work finally outgrows the CPU's launch time.
  - A 2048-token prefill is GPU-bound, but dominated by copies and elementwise ops on the length² score matrix,
    not by matmuls. That's the FlashAttention motivation, measured.
  - A unit test corrected my intuition: at batch 64 × 128 tokens, Qwen3-0.6B's KV cache (~0.94 GB) is nearly as
    big as its weights, so decode intensity is well below "≈ batch size".
- **Gate M1 approved** (questions deferred by the owner). Tagged `v0.1-nanoserve`.

- **M2 environment:** vLLM 0.30.0 pins PyTorch 2.13.0 (the research image has 2.14.0), so vLLM gets its own image
  and lock (`infra/serving.lock`) instead of changing the environment M0/M1 were measured in.
- **Dead end: slim base image.** vLLM loaded the model, compiled, captured CUDA graphs and sized the KV cache
  (172,640 tokens, close to the ~178k estimated in PROJECT.md Ch 4), then died in warm-up. FlashInfer compiles
  its top-k/top-p sampling kernel on first use and needs `nvcc`, which `debian_slim` lacks. Fix: build the serving
  image on `nvidia/cuda:13.0.1-devel-ubuntu24.04`.
- **Startup is slow (224 s):** the likely cause is FlashInfer recompiling every container start. Its cache now
  lives on the Volume (`FLASHINFER_WORKSPACE_BASE`), next to vLLM's compile cache (`VLLM_CACHE_ROOT`).
- **The fix worked:** with both compile caches on the Volume, the server now starts in about 95–100 s.
- **Library check:** `load_dataset("wikitext")` fails on current Hugging Face `datasets` ("Repository id must be
  'namespace/name'"). The dataset lives at `Salesforce/wikitext`.
- **Launcher bug:** the quality launcher lost every result when one (task, model) container failed. Each call is
  now collected on its own, and a failure is reported without discarding the others.
- **Reproducibility bug:** the needle grid seeded its random choices with `hash()`, which Python randomizes per
  process. It now uses `zlib.crc32`.
- **M2 speed results (run from `7a35017`):**
  - vLLM's offline engine is no faster than the served one, so HTTP and our client aren't the bottleneck.
  - **My 32k TTFT prediction left out the ×28 layers** in the attention FLOPs. Prefill actually runs at 68–83%
    of the M0 BF16 peak. FLOP counts now live in tested code (`report.m2.prefill_flops`).
  - **I first misread the peak.** I compared a KV-bound ceiling with the sweep's best whole-run average and
    reported "62% of the ceiling". Then, from client timestamps alone, I estimated a ~4,000 tok/s burst and
    called the peak half the real capacity. Both were wrong. The saturation run (a queue held for minutes, rates
    from vLLM's own `/metrics`) shows a sustained 1,849 tok/s for Qwen3-0.6B, 54% of the ceiling set by its live
    KV cache. **Lesson: capacity is a steady-state rate, read from the server's counters.**
  - **Prefill interference halves saturated decode.** With requests waiting, every step carries ~400 prompt
    tokens and runs at 53–54% of the memory speed. After the queue drains, the same engine runs at 86–88%.
    **Open question for M8:** those prompt tokens are little matmul work, so where does the extra time go?
    Candidates: mixed batches exceeding the CUDA-graph capture sizes, the mixed prefill/decode attention path,
    and CPU contention with the load generator on the same 8 vCPUs.
  - **Token-weighted context predicted the batch's context** (647 tokens per sequence; the server measured
    ~630). Long requests dominate the running set.
  - **Shared prefix is scheduler-bound:** each step's 2,048-token budget admits about one prompt, and Little's
    law in steps gives the batch size (~60 predicted, 56–58 measured, for both models).
  - vLLM 0.30 doesn't log `max_num_batched_tokens` in a form the log parser found; the timeline measures it
    instead (~2,050 tokens per step).
  - vLLM sizes the KV cache at every start: 142,928 then 152,800 tokens for the 1.7B on identical containers.
- **M2 quality results:** all 10 predictions within range. The needle test is saturated (100% for both models),
  so it can only catch breakage. KL and perplexity are the sensitive measures.
- **Gate M2 approved** (questions deferred by the owner). Tagged `v0.2-baselines`.
- **M3 environment:** `datasets` joins the research image for calibration text. The lock moved only `fsspec`
  (2026.9.0 → 2026.6.0, capped by `datasets`), a file-I/O helper that nothing measured in M0/M1 depends on.
- **Library check (llm-compressor 0.14.0, read from the installed source before comparing):**
  - GPTQ defaults to act-order "static", block 128 and dampening 0.01.
  - Its symmetric INT4 grid is `s = max|w| / 7.5`, with all 16 codes: 7% finer than the textbook `max|w| / 7`,
    at the price of clipping the largest positive weight by half a step. Our RTN now offers both (`full_range`).
  - AWQ moved: `llmcompressor.modifiers.awq.AWQModifier` is now a deprecated wrapper that builds a transform
    modifier plus a QuantizationModifier. The transform uses duo scaling (`s = x̄^α / w̄^(1−α)`), scores α on
    each parent module's output (the whole attention block for q/k/v), and has **no clipping search**. Our
    first AWQ scored concatenated linear outputs with the paper's `s = x̄^α`. It now follows the standard
    definition, with the paper's form and AutoAWQ's clipping as options.
  - llm-compressor quantizes all linears of a layer from one calibration pass (not "true sequential"); our
    comparison configuration matches that.
- **Wrong test, corrected:** I asserted that per-token FP8 activation scales beat per-tensor ones ≥10×, as they
  do for INT8. For FP8 they barely differ. A logarithmic grid has about the same *relative* precision at any
  scale. The test now states that contrast.
- **Worked example needed care:** my first hand-picked 4-weight GPTQ example had compensation that never crossed
  a rounding boundary, so GPTQ tied RTN exactly. A search over candidate weights (excluding near-ties, which
  float noise could flip) found one where RTN rounds two correlated weights the same way and GPTQ flips one.
- **llm-compressor's AWQ is slow here:** 17 minutes for Qwen3-0.6B vs 2.5 for its GPTQ. My dense-weight export
  afterwards hit a transient CUDA OOM warning (the library's caches still held the GPU); the allocator retried,
  and the export's own check passed (≤16 values per group of 128).
- **M3 results (Qwen3-0.6B, KL vs BF16 on WikiText-2):**
  - The pipeline checks out: BF16 nanoserve's perplexity matches M2's Hugging Face value. Our GPTQ and AWQ land
    within a few percent of llm-compressor's KL on the same grid and calibration tokens.
  - **Round-to-nearest is harsher than I predicted at every width.** The model has massive activations
    (residual channels ~1,300× the median). A weight column that reads one turns ordinary rounding error into
    huge output error. AWQ, which protects those columns, beats GPTQ at 4 bits. The most fragile module is layer
    2's down_proj, which writes the massive channels: the residual stream jumps from layer 3 onward.
  - **Rotation made RTN worse; the cause was folding, not rotating.** A follow-up run separated the steps.
    Folding the RMSNorm γ into the weights alone makes RTN INT4 3× worse: one norm's γ spans 61× between its
    largest and median channel. Rotating afterwards helps, but not back to plain RTN. BF16 rounding of the
    rotated weights costs almost nothing. With GPTQ, rotation gives the best INT4 result of the milestone.
  - Asymmetric grids matter far more than expected (INT3: 4.5 → 1.9 KL). NF4 beats uniform INT4 at the same block.
  - FP8 W8A8 is insensitive to how its activation scale is chosen. INT8 needs per-token scales or SmoothQuant.
  - Calibration: 8 sequences ≈ 128. Domain matters: code calibration makes GPTQ no better than RTN on prose.
  - INT8 *lowered* Qwen3-1.7B's perplexity below BF16 while its KL stayed at 0.006: perplexity can reward
    damage. KL is the honest measure.
- **Missing log fixed:** PROJECT.md promised a `results/compute_log.csv` from M0 on, and it was never written.
  It now comes from Modal's own billing API (`scripts/compute_log.py`), not from estimates.

## 2026-10-01

- **Gate M3 approved** (questions deferred by the owner). Tagged `v0.3-quant-reference`.
- **Library check for M4 (llm-compressor 0.14.0):**
  - FP8_DYNAMIC means FP8 per-channel weights with dynamic per-token FP8 activations. W8A8 is the INT8
    equivalent. Both match M3's simulated `w8a8-*-token` configurations.
  - SmoothQuant now lives among the transform modifiers.
  - `oneshot` shuffles calibration samples by default. That doesn't affect M3: GPTQ's Hessian and AWQ's
    statistics are sums over samples.
- **Dead end: AWQ out of GPU memory.** llm-compressor's AWQ caches every parent module's inputs and outputs for
  all 128 × 2,048 calibration tokens on the GPU. Qwen3-0.6B's AWQ checkpoint ran out of the L4's 22 GiB (M3's
  run had only just fit, with OOM warnings). The laptop couldn't even show the error: the remote exception is a
  torch type, and the local venv has no torch. The fix keeps the calibration identical: `AWQModifier(offload_device="cpu")`,
  with more container RAM.
- **Not done: publishing checkpoints.** AGENTS.md asks for Hugging Face uploads with model cards. Publishing is
  outward-facing and needs the owner's account, so the checkpoints stay on the Modal Volume until they decide.
