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
