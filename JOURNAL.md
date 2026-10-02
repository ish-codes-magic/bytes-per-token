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
- **Dead end: servers couldn't see new checkpoints.** Half the quantized servers failed with vLLM's
  `validate_repo_id` on `/cache/m4/...` paths that existed. Modal reuses warm containers, and a reused
  container keeps its old view of the Volume, from before another container committed the checkpoint. So vLLM
  saw no directory and treated the path as a Hub repo id. Every function that reads checkpoints now calls
  `hf_cache.reload()` first, and the reruns passed.
- **M4 results (vLLM 0.30.0, L4):**
  - Batch-1 decode: FP8 1.33× / 1.46× and INT4 1.69× / 2.18× over BF16 (0.6B / 1.7B). A bytes-only model
    (bytes ÷ M0 bandwidth + a fixed overhead fitted on BF16) predicts INT4 within 5%. W8A8 runs 6–9% slower
    than its bytes predict.
  - **Why W8A8 lags:** read vLLM's installed source. `CutlassInt8ScaledMMLinearKernel.apply_weights` runs
    `ops.scaled_int8_quant(x)` as its own op before `ops.cutlass_scaled_mm`. Dynamic per-token scales need a
    pass over each activation first. That's M7's fused RMSNorm+quant kernel, now with a measured motivation.
  - Saturation barely moves (cost changes between −6% and +1%), as M2 predicted: the KV cache dominates.
  - The INT4/FP8 crossover falls between batch 64 and 256 on both models.
  - **My own mistake:** the learning doc's section 2 called B* ≈ 56 the crossover. B* is where INT4's matmuls
    turn compute-bound. With the linear layers alone, INT4 stays ahead until its 16-bit math time equals FP8's
    streaming time, at batch ≈ 109. Corrected in the results section; the pre-registered prediction is left as
    written.
  - **Kernel fidelity:** vLLM's own perplexity (from prompt logprobs) matches nanoserve's simulation of the
    same rounded weights within 1% for every format (AWQ worst, ~0.8%). The low-bit kernels compute what the
    checkpoints say. *(Corrected below: the AWQ gap was the dense export, not the kernel.)*
  - Quantized formats get more non-KV memory reserved at startup, so the KV cache gains less than the weights
    free. Cause not found (the log only has totals). Open for M5.
- **Surprise: GPTQ's GSM8K score is partly a stopping failure.** INT4 GSM8K dropped 29 / 22 points (0.6B / 1.7B),
  far outside the predicted −15…−3 / −10…−1. Logging every answer showed that the GPTQ checkpoints talk past
  `#### N` on ~40% / ~37% of problems (BF16 ~1.5%, AWQ 3% / 9%). The suite's flexible-extract metric scores the
  last number said, so a right answer followed by chatter counts as wrong. On strict-match, 1.7B GPTQ is −10
  (just outside the prediction) and beats AWQ, as its lower KL predicted. 0.6B's INT4 loss is real on either
  metric: wrong reasoning and loops. Why GPTQ and not AWQ? Unknown; one hypothesis is GPTQ's error compensation
  fitting C4 (no few-shot Q&A in it). Calibrating on GSM8K-style text would test it. Not done: out of M4's
  scope.
- **Spend:** September used $15.28 of Modal's $30 monthly credit (billing through 30 Sep 21:00 UTC). M4's
  checkpoints (64 GB containers) and ten servers cost the most. The later reruns, fidelity check and GSM8K
  logging aren't billed yet. None of the owner's $10 reserve is used.
- **Gate M4 approved** (questions deferred). Tagged `v0.4-quant-production`. The owner approved publishing
  the checkpoints and freeing the Modal Volume.
- **Correction: the AWQ fidelity gap was llm-compressor's dense export, not vLLM.** Before deleting the
  dense exports, I wrote our own reader for compressed-tensors checkpoints (`quant/compressed.py`; int4
  unpacking checked against the library's `pack_to_int32` layout) and compared it against them. FP8 and
  GPTQ match exactly. INT8 and AWQ differ in every linear: 0.16% / 0.6% of weights by one grid step. Those
  are the two recipes that rescale weights first; the export re-rounds BF16 weights that sit near rounding
  boundaries, while the checkpoint stores the original codes. vLLM serves the codes. Re-measuring M4's KL on
  the codes moved the fidelity gaps to ≤0.2% for every format (AWQ 0.9% → 0.0% on 0.6B). AWQ KL moved
  slightly (0.227 → 0.232 on 0.6B); conclusions unchanged.
- **Dead end in that diagnosis:** I first measured how far each copy sat from the stored grid. Both were
  equally off, because the BF16 product of code and scale rounds. Counting grid steps between the two copies
  was the measure that worked.
- **Started M5 (KV-cache engineering).** Predictions committed in `b93afc6` before any measurement.
- **vLLM 0.30 API notes (read from the installed source):**
  - The attention backend is chosen with `--attention-backend FLASHINFER`; there is no
    `VLLM_ATTENTION_BACKEND` environment variable in this version.
  - FP8 KV defaults to a per-tensor scale of 1.0.
  - On the L4 FP8 KV forces FlashInfer: FlashAttention's FP8 path needs FA3 on Hopper. So every FP8-KV
    comparison needs a BF16-KV-on-FlashInfer control, now in the config.
  - Prefix-cache counters are `vllm:prefix_cache_{queries,hits}`. Per-request cached tokens need
    `--enable-prompt-tokens-details`.
- **What capped M4's saturated batch:** every saturated run peaked at exactly 256 running: vLLM's default
  `max_num_seqs`, not the cache. The cache also hit 100% at times, with preemptions (42 on 0.6B, 363 on 1.7B).
  So at saturation FP8 KV can't raise the batch, only halve the KV bytes per step. A new `capacity` workload
  (96 users × 4k-token prompts) isolates the capacity effect.
- **A false-hit trap avoided:** the new multi-turn workload with seed 0 would have started with the same
  random tokens as M2's shared-prefix workload, giving vLLM cross-workload cache hits. It has its own seed,
  and each prefix-caching server runs one load per workload, so no load replays requests into a warm cache.
- **Surprise: Qwen3's keys have huge fixed outlier channels.** In layer 0, KV head 0, key channel 50 reaches
  |452|, with a median channel max of 5.8. That's just past FP8 E4M3's largest value (448), so at vLLM's
  default scale of 1.0 that channel saturates. QK-norm normalizes each key head's RMS, yet a few channels
  still dominate; whether its learned per-channel weight or RoPE puts them there is not yet checked.
  - Consequence: per-token INT8 KV cost KL 0.019, *more* than FP8's 0.015 (predicted 0.0002–0.003). One
    channel at 452 stretches each token's grid to steps of ~2 for channels near 5.
  - **Control added after seeing this (not a prediction):** INT8 with per-channel keys (KIVI-style) costs KL
    0.0014, 13× less. The cause is the outlier channels, not the bit width.
  - INT4 per-token keys are catastrophic (KL 6.0, top-1 10%); rotating the keys helps (1.0) but isn't
    enough when one channel holds most of the vector's energy; KIVI's per-channel keys reach 0.032.
- **Correction:** the |452| above is KV head 0's peak. Over all heads, layer 0's largest key on Qwen3-0.6B is
  |506|. Qwen3-1.7B's peaks at |394|, inside FP8's range.
- **M5 results (vLLM 0.30.0, L4; nanoserve for the KV policies):**
  - 22 of 31 predictions in range. Capacity, 32k latency, needle recall and prefix caching behaved as modeled.
  - **The kernel confound was real and large.** BF16 KV on FlashInfer is 1.45× (0.6B) / 1.24× (1.7B) faster
    than on FlashAttention at saturation, with the same ~250 sequences and a full cache. FP8 KV's 2.19× is
    that kernel switch × FP8 storage (1.52×). Without the control I would have credited FP8 with all of it.
  - **That revises M2's conclusion.** M2 blamed the gap to the memory-bound ceiling (54%) on chunked-prefill
    tokens in every step. The same mixed steps run at 79% of memory speed on FlashInfer. A large part of the
    gap is FlashAttention's handling of mixed batches on Ada, not the mixing itself. Why: M7's profiler.
  - FP8 KV: KV tokens exactly 2× on the same kernel (FlashInfer reserves more memory than FlashAttention,
    hence 1.83× vs FlashAttention on 1.7B). Capacity workload: 39 → 70 running, 1.77× throughput. 32k decode
    1.47× faster; TTFT unchanged. Needle 100%, perplexity +1.2% (0.6B) and −2% (1.7B, the "perplexity
    rewards damage" trap again; KL 0.012).
  - FP8 weights on top of FP8 KV make the saturated 0.6B server 12% slower (W8A8's activation pass, as in M4)
    but help 1.7B by 8%.
  - Prefix caching: vLLM cached 85.3% of multi-turn prompt tokens against the radix tree's 85.5% ceiling.
    TTFT 222 → 60 ms; throughput 1.68× (above range) because shorter prefill chunks also shorten everyone
    else's decode steps (TPOT 18 → 12 ms).
  - StreamingLLM's KL (0.038) was *below* the predicted range while its needle score was 32%: WikiText barely
    uses context beyond 1,000 tokens. KL alone would have called eviction safe.
- **Cost:** M5's campaign ran ~22 containers in parallel. Billing will show it under October's credit.
- **Gate M5 approved** (questions deferred). Tagged `v0.5-kv-cache`. Started M6 (speculative decoding);
  predictions committed in `995a42e` before any measurement.
- **vLLM 0.30 speculative decoding, read from the installed source:** `--speculative-config` JSON with
  `method` in {draft_model, ngram, eagle3, suffix, ...}, `num_speculative_tokens`, and for n-gram
  `prompt_lookup_max/min`. It also has a built-in draft-length schedule by batch size
  (`num_speculative_tokens_per_batch_size`). Counters: `vllm:spec_decode_num_{drafts,draft_tokens,
  accepted_tokens}` and `..._accepted_tokens_per_pos` (labeled by position). A public EAGLE-3 head exists
  for our target: `AngelSlim/Qwen3-1.7B_eagle3`.
- **Design: one interface, `logits_after(context, last)`.** A cached nanoserve model and a toy bigram table
  both implement it, so the same draft-verify loop runs on real models and on a model whose exact output
  distribution is known. KV "rollback" after a rejection is just overwriting from the first differing
  position: stale slots sit past the current position, where the causal mask can't see them.
- **Insight that made the experiment cheap:** under greedy decoding, while a draft matches, the drafter sees
  the target's own context. One teacher-forced drafter pass over the target's output gives a bit per
  position, and the speculative run for *every* k follows from the bits. A test checks the replay equals the
  real loop's rounds on tiny models.
- **Dead end: a negative control that was too weak.** The first "lying drafter" reported a uniform q. On the
  toy model the chi-square test caught it, but on tiny Qwen3 models with 1,000 samples it slipped under the 5σ
  limit (74 vs 103). A weak control proves little. It now reports q ≈ 0, so every draft token is accepted
  and the output follows the *drafter*: caught decisively.
- **Dead end: the EAGLE-3 head wouldn't load.** "Cannot find any model weights": my download helper fetches
  only `*.safetensors` and config files, and the head ships its weights in another format. Drafters now get
  their whole repo.
- **Surprise: nanoserve's drafter step costs as much as its target step** (45 ms vs 46 ms at a 621-token
  context). Both models have 28 layers, and the reference engine at batch 1 is bound by per-layer overhead
  (Python, kernel launches), not by streaming weights. So "c" is ~1 in nanoserve, and draft-model speculation
  can't pay there. The memory-bound claim that *is* visible: a target pass over 4 new tokens costs 1.05× a
  pass over 1 (9 tokens: 1.09×). Speed is measured in vLLM.
- **Surprise: the real loop's greedy output matched plain greedy on only 5 of 8 prompts** (BF16), diverging
  at tokens 3, 31 and 38. Hypothesis: a k+1-token pass and a 1-token pass round differently in BF16, which
  flips near-ties. Added a float32 control to test that, instead of assuming it.

## 2026-10-02

- **M6 results (nanoserve for agreement and losslessness; vLLM 0.30.0 on the L4 for speed):**
  - 17 of 28 predictions in range. The misses: the drafter is better than I guessed; BF16 isn't exact; costs
    inside vLLM's speculative loop differ from standalone; a busy server hurts less than my worst case.
  - **Agreement:** Qwen3-0.6B predicts 83% of Qwen3-1.7B's greedy tokens (code 95%, math 90%, chat and
    summarization 72%). The replay's 2.99 tokens per pass at k = 3 matches vLLM's counters (3.0).
  - **Not independent:** agreement is 85% after an agreement and 72% after a miss. Rounds start after a miss,
    so tokens per pass sit ~3–7% *below* E(α, k). (My first caption said "because of runs", the wrong way
    round. Fixed.)
  - **One user:** draft model 1.21× / 1.24× / 1.20× at k = 1 / 3 / 5; EAGLE-3 head 1.70× with only 38% of
    its tokens accepted; n-gram 1.7× on code and 0.86–0.91× on chat.
  - **Why EAGLE wins:** a drafted token costs it 1.2 ms (c = 0.08) against 6.3 ms (c = 0.43) for the small
    model. Cheap guesses beat good guesses.
  - **Busy server:** at 64 users the draft model gives 0.69–0.88×, n-gram 0.92–0.97×, EAGLE still 1.21×.
  - **Hidden cost:** a draft model cuts the KV cache from 152,800 to ~64,000 tokens (its own weights and
    its own KV for every token). EAGLE and n-gram cost ~10%.
  - **Interaction:** the drafter's agreement barely moves with a quantized target (83.1% → 83.1% FP8, 82.0%
    INT4), but the speedup goes 1.24× → 0.79× (FP8) → 0.50× (INT4): the target's step shrinks, the drafter's
    doesn't. Speculation and quantization spend the same slack.
- **Open: the INT4 drafter is slower than the BF16 drafter inside vLLM's speculative loop** (7.4 ms vs 6.3 ms
  per drafted token; standalone 3.4 ms vs 5.8 ms). vLLM did select Marlin for it. Cause not found. A
  candidate for M7's profiling.
- **Surprise that became a finding: BF16 logits depend on the shape of the pass.**
  - Greedy: the real loop matched plain decoding on 11 of 16 prompts in BF16 and 16 of 16 in float32. vLLM's
    outputs with a drafter were byte-identical to no speculation for only 44% of requests.
  - Sampling: the first run showed plain sampling *failing* the chi-square test against "its own" exact
    distribution (339 vs a limit of 12) on the chat prompt. Impossible for a correct sampler, so I checked
    the reference instead of the sampler. The reference pass (one sequence, no cache), the plain sampler's
    pass (a batch of 50 through a cache) and the verifier's pass (4 new tokens on a cached prompt) put the
    first token's distribution up to 0.23 apart in total variation in BF16, and 10⁻⁵ apart in float32.
  - Each sampler passes against the distribution its own pass computes, in both dtypes.
  - Lesson: when a "can't happen" result appears, question the yardstick first. And "lossless" has to be
    stated against a specific computation, because BF16 doesn't have one true distribution.
- **A borderline result checked, not waved through:** float32 speculative sampling scored χ² = 11.8 at a
  limit of 12 with 600 samples. With 3,000 samples it scored 2.6. A bias would have grown fivefold.

- **M7 started. What vLLM already has, read from the installed source before designing anything:**
  - A fused RMSNorm + quantization op exists (`rms_norm_dynamic_per_token_quant`) and a compile pass that
    rewrites norm → quant into it (`rms_quant_fusion.py`). Its tables hold FP8 keys only. For INT8
    checkpoints the two ops always run apart. So kernel 1 targets INT8, not FP8 as AGENTS.md first suggests.
  - No attention backend reads an INT4 or INT8 KV cache. FlashInfer exposes a single-request decode call
    (`single_decode_with_kv_cache`), which gives kernel 2 a production kernel to be compared with.
  - API note: the serving image has Triton 3.7.1, the research image 3.8.0. Both kernels run on both.
- **Order of work, stated honestly:** both kernels and their references were written and tested for
  correctness before the formal profile ran. No kernel was *timed* before the profile was recorded and the
  predictions committed (`eb13c86`).
- **Profile findings:**
  - nanoserve's decode step at 32k tokens: attention is 71% of it and reads the cache at 14% of the
    bandwidth. At 512 tokens the step is launch-bound and attention is 15%.
  - vLLM's `rms_norm` and `scaled_int8_quant` each cost ~1.6 µs in a CUDA graph at decode sizes and run at
    ~235 GB/s at 32,768 tokens. vLLM's own fused FP8 op is 3.5–4.4× slower than its two INT8 ops at
    4,096 tokens.
  - **M4 was partly wrong.** M4 blamed W8A8-INT8's unexplained batch-1 gap on the separate quantization op.
    112 launches × 1.6 µs = 0.18 ms, against a gap of 0.37 ms (0.6B) and 0.62 ms (1.7B): 47% and 29%. I had
    drafted a prediction about this, then saw the profile already answered it and removed the prediction
    rather than keep one I knew the answer to. M4's docs now carry a correction.
- **M7 results. 21 of 29 predictions in range.**
  - **Kernel 1** vs vLLM's two ops in a CUDA graph: 1.9× for one token, 2.5× at 4,096 tokens, 2.3× at
    32,768 (90% of the bandwidth). From Python: 0.95× for one token (Triton's launcher costs what two native
    calls cost). 96.6–97.8% of its codes equal vLLM's, never more than one apart: vLLM rounds the norm's
    output to BF16 first.
  - **Kernel 2**, one layer, 1 × 32,768 tokens: PyTorch's path 3.2 ms; FlashInfer (FP16) 0.54 ms; kernel 2
    on BF16 0.58 ms, INT8 0.32 ms, INT4 0.21 ms. So INT4 codes are 2.6× FlashInfer on full precision, and
    on the same bytes FlashInfer wins (0.93×). At 512 tokens kernel 2 loses to FlashInfer (0.44×).
  - **nanoserve**, 1 × 32,000 tokens: 139 → 54 ms per step (2.6×), identical for BF16, INT8 and INT4 caches.
    At 512 and 4,096 tokens the new cache is slower (0.7×).
  - **Quality through the real kernel:** KL 0.0011 (BF16 read by the kernel: the yardstick's floor), 0.0012
    (INT8), 0.011 (INT4); needle recall 75 of 75 with INT4.
- **Surprise: the best split is tiny.** Predicted 512–2,048 tokens per program; measured 128, with one warp.
  More programs keep more loads in flight, and one warp keeps a program's sums inside the warp. Defaults
  were set from the first sweep.
- **Surprise that became a methodology fix: "cold" timings were biased.**
  - Symptom: cold timings of the BF16 kernel at 32k tokens (128 MiB, far beyond L2) were ~150 µs *slower*
    than a warm CUDA-graph replay. Warm cannot help data that doesn't fit in cache.
  - Cause: `cuda_time_ms(flush_l2_bytes=…)` evicts L2 by overwriting a scratch buffer. The cache is then
    full of modified lines, and the timed call pays to write them back.
  - Fix: evict by reading (`flush_by="read"`). FlashInfer at 32k went from 691 to 539 µs.
  - Check: vLLM's short step + 28 × FlashInfer's layer time = 20.9 ms with the read flush (vLLM measured
    20.6) and 25.1 ms with the write flush. The read flush is the right one.
  - Consequence: M0's cold matmuls and M1's decode points used the write flush. Not re-measured (outside
    M7); proposed for M8, before the performance model is fitted.
  - The attention benchmark and the tuning sweep were rerun with the read flush. The first runs stay in
    `results/raw/m7_kernels.jsonl`.
- **Surprise: the cold timing hides Python.** The flush keeps the GPU busy while Python queues the next
  call, so a "cold" number is pure GPU time. From Python, kernel 2 costs ~210 µs per call at any short
  context (launcher + the merge's small ops) against PyTorch's ~150 µs. That is why the kernel "wins" at
  512 tokens on the cold clock (1.3×) and nanoserve still gets slower there.
- **Finding: INT4 is instruction-bound.** At 1 × 32,768 the INT4 cache (37 MiB) fits in L2, and the warm
  graph replay takes 201 µs against 205 µs cold. Warm at 8,192 tokens, BF16 and INT4 both take 65 µs.
  BF16 and INT8 are memory-bound at 86–94% of the bandwidth; INT4 tops out near 72%.
- **Finding: after fusion nanoserve is bound by Python.** At 16k tokens the GPU works 69 of 72 ms before
  and 12 of 73 ms after (under the profiler), in 2,582 kernels. The old path's time was three copies of the
  cache: an indexing kernel reading it out (14 ms), elementwise copies to give each query head its own K
  and V (25 ms), then PyTorch's attention kernel (24 ms).
- **Lost run:** the first nanoserve task died at the timeline step. A decode step created under
  `inference_mode` was called by the profiler after its maker returned, so its tokens met autograd.
  Nothing was recorded. The step function now carries the decorator itself.
- **Provenance note:** the second nanoserve run was launched with the figure module uncommitted and is
  flagged dirty. It was rerun from a clean tree; the reports use the newest records.
- **Dead ends and things not done:**
  - Tensor cores for INT4's inner products (`tl.dot`): needs half-precision operands, and the scaled
    queries would lose precision. Not attempted. It is the next step for the instruction bound.
  - A fused merge kernel, to cut the ~27 µs fixed cost that loses to FlashInfer at short context.
  - vLLM integration of either kernel. The path is written down in docs/04-kernels.md; the projection there
    is labeled as one.
- **Still open from earlier:** the rest of M4's INT8 gap; the INT4 drafter that is slower inside vLLM's
  speculative loop (M6), which M7's profiling did not reach.

### M8: the full stack

- **M7 gate decisions (from the human).** Custom kernels are reported as a nanoserve measurement plus a
  labeled vLLM projection; no vLLM backend is written. M0's and M1's cold timings are not re-measured: the
  write-flush bias stays documented where those numbers are used. Tagged `v0.7-custom-kernels`.
- **Checkpoints published.** Eight repos under `ishita-codes-ai/` (FP8-Dynamic, W8A8-INT8, W4A16-GPTQ,
  W4A16-AWQ, for both models), each file's hash checked against the Volume copy before that copy was
  deleted. The M4–M6 configs now name the Hub ids. The token is read inside the container from the Modal
  secret `huggingface-secret` and is never printed or returned.
- **Caught by the startup check: a server that could not answer.** The first published-checkpoint server
  returned HTTP 400 on a chat request. The loader's filtered download fetched only the files nanoserve reads
  and skipped `chat_template.jinja`. Servers now take the whole snapshot (`checkpoint_dir`). Nothing had
  been timed.
- **Plan.** Techniques are letters: `w` FP8 weights, `a` INT4 (AWQ) weights, `f` FlashInfer on a BF16 cache
  (control), `k` FP8 KV, `p` prefix caching, `s` speculative decoding (EAGLE-3 head, k = 3). A label is the
  letters that are on. Qwen3-1.7B runs the full 2⁴ factorial of w, k, p, s (ladder, leave-one-out and every
  pair in one design), plus `wf`, `a`, `akps`, and repeats of `base` and `wkps`. Qwen3-0.6B runs the ladder
  and leave-one-out. 30 servers, five workloads.
- **The serving model was built and frozen first** (`perfmodel/serving.py`, calibrated on M2–M6 only).
  - 182 earlier points: median error 3.4%, 94% within 15%. 101 of them were not used to fit: 4.5%.
  - Not modeled: servers with a separate draft model (42 points, 55% off).
  - What fitting it taught, each one a mistake first:
    - Tokens per target pass must count every pass, including those that accept nothing: G ÷ (G − A).
    - A prefill that rides in a decode step does not pay for its own read of the weights.
    - A closed-loop client's "64 users" are not 64 running sequences; in-flight requests are measured.
    - A prefix shared by a batch is read from L2 after the first sequence, if one layer's slice fits.
  - Predictions for all 30 servers: `benchmarks/predictions/m8_model.json`; 32 hand-ranged claims in
    `m8.json`. Both committed before any server ran.
- **Pilot (`1.7b-base`).** Latency 67 tok/s, busy 1,955, capacity 256, multi-turn 199, long 10. GPU power
  is ≈ 72 W (the L4's limit) on every workload, one user included: a decode step keeps the GPU at its
  power limit even when it is far from its FLOP limit.
- **All 30 servers ran; none failed.** Quality of `wk` (the lossy part of the stack) measured in vLLM.
- **Surprise: the full stack is not the best stack.** On Qwen3-1.7B at one user: `ws` 151 tok/s, `wkps` 76
  (base 67). On Qwen3-0.6B the full stack is *slower than stock*: 80 tok/s against 168.
  - Every server with both `k` and `s` is slow, at one user by 25–75%: `ks` 85 against `s` 113 and `k` 65.
  - The serving model is within ~13% on every 1.7B server without that pair, and 35–110% high with it.
  - Tokens kept per pass are the same with and without `k` (2.1): the drafter is not worse.
  - GPU power on those servers falls to 46–58 W. The GPU is waiting for the host.
- **Hypothesis, from vLLM's startup log.** `--kv-cache-dtype fp8` forces FlashInfer on an L4 (M5). With
  speculation, vLLM 0.30 prints: "CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
  attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting
  cudagraph_mode=PIECEWISE". So `ks` runs without a full CUDA graph: every attention call is made from
  Python. This is the third lever in reverse: waste between bytes and math.
- **Controls added to the plan, predictions first** (`benchmarks/predictions/m8_controls.json`):
  - `fs`: FlashInfer with speculation on a BF16 cache. If the FP8 bytes are innocent, `fs` ≈ `ks`.
  - `g`, `sg`: FlashAttention with piecewise graphs only. If the graph mode is the whole cause,
    `sg` ÷ `s` ≈ 0.75. I expect 0.84–0.96: losing the graph costs some host time per layer, and the rest is
    FlashInfer's own multi-token path (it treats a 4-token decode as a prefill and plans it on the host).
  - `aps`: the INT4 branch without the collision, the candidate for the best one-user stack.
  - Server starts now record which CUDA graphs were captured (`cuda_graphs` in `server_start`).
