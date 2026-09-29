# ADR 001: Cloud-only compute with small models on an L4

- **Status:** accepted
- **Date:** 2026-09-29

## Context

The original plan (AGENTS.md §3) targets Llama-3.1-8B on H100 GPUs. Three constraints rule that out:

1. **Budget:** about $10 of personal money for the whole project.
2. **Local disk:** the development laptop can't hold model weights, CUDA toolchains or container images (tens of GB).
3. **Resource use:** the owner wants small instances, not large ones.

## Decision

- **All compute runs in the cloud.** GPU work runs on **Modal** (serverless, billed per second, $30/month of free
  credit on the Starter plan). CPU tests run on Modal or GitHub Actions. The laptop holds the repository plus the
  `modal` client and `ruff` (~50 MB).
- **One GPU type for every recorded speed number: the NVIDIA L4** (Ada, 24 GB, ~300 GB/s, BF16 and FP8 tensor
  cores, 72 W, ~$0.80/h).
- **Models: Qwen3-0.6B (main) and Qwen3-1.7B.** Qwen3 is AGENTS.md's fallback family. Both are ungated and share a
  tokenizer (so 0.6B can draft for 1.7B). Their layer widths are multiples of 128, as production INT4 kernels require.
- Unit tests use tiny random-weight models, so they never download anything.

## Consequences

- The physics is preserved: Qwen3-0.6B on an L4 needs ~4 ms per memory pass, the same memory-bound regime as an 8B
  model on an H100 (~5 ms). All three levers stay visible.
- No 8B-scale results. Claims are about relative gains and their causes; the performance model (M8) predicts scaling.
- Nsight Compute is likely unavailable in Modal's containers, so kernel analysis relies on measured bandwidth against
  the M0 roofline.
- The Dockerfile from AGENTS.md M0 is deferred to M9. Until then, the Modal image (built from the same `uv.lock`) is
  the reproducible environment.

## Alternatives considered

| Option | Why not |
|---|---|
| H100 + Llama-3.1-8B (original plan) | ~$150–300 of GPU time |
| Laptop RTX 3050 (4 GB) + Llama-3.2-1B | Needs ~30 GB of local disk for models, CUDA and envs |
| T4 instead of L4 | No BF16 or FP8, and weaker vLLM/Triton support |
| SmolLM2-135M/360M | Hidden sizes 576 and 960 are not multiples of 128, which breaks INT4 kernels |
| Llama-3.2-1B | Gated, larger, and has no smaller sibling to act as a drafter |
