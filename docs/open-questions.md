# Open questions, kept runnable

Things the project measured but did not explain. Each has the exact command that would most likely settle it,
what it costs, and how to read the result. Nothing here is needed to reproduce the results.

## 1. Why do FP8 weights double the host's time per step on piecewise CUDA graphs?

**What is known** ([docs/06-results-analysis.md](06-results-analysis.md#66-round-4-what-sets-the-hosts-time),
section 6):

<!-- BEGIN GENERATED: m8_controls_4 -->
*Predictions written in commit `2069f67`, before the first measurement.*

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| Qwen3-0.6B, BF16 weights, piecewise graphs only: ms per step (full graph: 5.9; if FP8 is what lengthens the chain, well below wg's 25.8) | 6 – 15 | 12.1 | within range |
| Qwen3-1.7B, FP8 weights, piecewise graphs only: ms per step (full graph: 10.3; if FP8 lengthens the chain, near 25) | 22 – 28 | 25.3 | within range |
<!-- END GENERATED: m8_controls_4 -->

On piecewise graphs a step waits for the host, and the host needs about twice as long with FP8 weights as
with BF16 weights, on either model. The GPU does the same work, and the number of launches per step is the
same with either format:

<!-- BEGIN GENERATED: m8_profiles_small -->
| Server | CUDA graphs | Wall time per step, profiled (ms) | GPU kernel time per step (ms) | GPU busy | Graph launches per step | Kernel launches per step, outside graphs |
|---|---|---|---|---|---|---|
| `w` | full | 6.6 | 4.3 | 65% | 1 | 10 |
| `g` | piecewise | 28.4 | 5.5 | 19% | 29 | 66 |
| `wg` | piecewise | 39.8 | 4.2 | 11% | 29 | 66 |
<!-- END GENERATED: m8_profiles_small -->

**What the existing profiles point at.** The two small-model profiles, call by call
(`scripts/compare_profiles.py 0.6b-wg 0.6b-g`):

<!-- BEGIN GENERATED: m8_profile_difference -->
| Difference (ms per step) | `wg`: self time × calls per step | `g` | Kind | Call |
|---|---|---|---|---|
| +27.37 | 27.37 ms × 0.1 | not among the rows kept | python_function | `threading.py(359): wait` |
| +23.21 | 49.56 ms × 6.2 | 26.35 ms × 6.0 | python_function | `<built-in method acquire of _thread.lock>` |
| +11.36 | 39.77 ms × 0.0 | 28.41 ms × 0.0 | python_function | `vllm/usage/usage_lib.py(255): _report_continuous_usage` |
| +10.95 | 34.34 ms × 1.0 | 23.40 ms × 1.0 | python_function | `vllm/v1/worker/gpu_worker.py(1147): execute_model` |
| +4.78 | 4.78 ms × 28.0 | not among the rows kept | python_function | `vllm/v1/attention/backends/flash_attn.py(272): __call__` |
| +4.04 | 4.48 ms × 56.0 | 0.45 ms × 56.0 | python_function | `torch/_tensor.py(1044): split` |
| +0.55 | 1.69 ms × 0.0 | 1.14 ms × 0.0 | python_function | `vllm/v1/worker/gpu_worker.py(1209): execute_model` |
| +0.53 | 2.06 ms × 0.1 | 1.54 ms × 0.1 | python_function | `threading.py(355): wait` |
| +0.48 | 2.27 ms × 28.0 | 1.79 ms × 28.0 | python_function | `vllm/v1/attention/backends/flash_attn.py(1166): forward` |
| +0.48 | 0.48 ms × 84.0 | not among the rows kept | python_function | `<built-in method expand of Parameter>` |
<!-- END GENERATED: m8_profile_difference -->

- Trivial Python calls (`Tensor.split`, the attention wrapper's `__call__`) take several times longer per
  call with FP8 weights, with the same number of calls.
- A large part of the difference is self time of `execute_model` that no child call accounts for, and time
  other threads spend waiting on a lock.
- Read together: the main thread is being slowed down across the board, as if something else holds the
  interpreter or the CPU. That is a lead, not a finding. These profiles kept only the 40 largest rows, and
  the profiler with Python stacks distorts exactly the code that is under suspicion.

**The run that would most likely settle it**

```bash
make open-question
# = uv run --only-group local modal run infra/modal_app.py::m8 --profile --only 1.7b-wg,1.7b-g
#   uv run --only-group local python scripts/compare_profiles.py 1.7b-wg 1.7b-g
```

- **What it does:** profiles the 1.7B model on piecewise graphs with FP8 and with BF16 weights (24 engine
  steps each, Python stacks on, the 150 largest rows kept), then prints the calls whose host time per step
  differs most.
- **Cost:** two L4 containers for about six minutes each, billed by Modal.
- **Why the 1.7B model:** it separates "FP8 weights" from "the small model", as round 4 did for the timing.
- **How to read it:**
  - one or two calls carry the difference → that is the code path. Read it in vLLM's source with
    `serving_library_facts("source:vllm/<path>")` (infra/modal_app.py).
  - every call is slower by a similar factor, and a lock or `wait` row grows → another thread is competing.
    Rerun with `--no-stack` (`::m8 --profile --no-stack --only 1.7b-wg`) to see whether the slowdown
    survives without the Python tracer; if it does, look at what the FP8 path starts in the background.
  - no difference under the profiler → the profiler hides it; time the steps instead with the two servers'
    step counters (`step_ms` in src/fastserve/report/m8.py), which is how round 4 measured it.
- **Then:** write the prediction first (benchmarks/predictions/), append the result to `JOURNAL.md`, and
  update section 6.8 of the results analysis.

## 2. Is FlashInfer's loss at 64 users the same mechanism?

With FP8 weights, FlashInfer servers at 64 users land between the full-graph server and the forced-piecewise
control ([06, section 6.7](06-results-analysis.md#67-every-server-on-piecewise-graphs)). That fits "busy
passes carry a prompt, so they run outside the full graph", but vLLM does not log which graph a pass used.

**The run:** a server profile at 64 users instead of one. `infra/modal_app.py::m8_profile` sends a single
request; it would need a second mode that drives the `spec_mixed` workload while profiling, and then
`cudaGraphLaunch` per step says how many passes were replayed whole. Not built. About the same cost as
question 1.

## 3. Does the collision exist on Hopper?

Read from vLLM 0.30's source, it should not: with FlashInfer's TRTLLM kernels a speculative pass stays in a
full graph, and FlashAttention has its own FP8 path there. Not measured: this project has no H100 time.

**The run:** `1.7b-ks`, `1.7b-s` and `1.7b-k` from `benchmarks/configs/m8_ablation.yaml` on an H100, and
the `cuda_graphs` field of their `server_start` records. If `ks` captured a full graph and its interaction
is near 1, the collision is specific to older GPUs.
