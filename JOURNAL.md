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
