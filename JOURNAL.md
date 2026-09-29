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
