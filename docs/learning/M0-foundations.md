# M0: Foundations: what can this GPU really do?

Every later result in this project is judged against two numbers: how fast the GPU can **move bytes** and how fast
it can **do math**. Datasheets promise both. M0 *measures* them on our GPU, an NVIDIA L4, because promised numbers
are never reached in practice.

## 1. Intuition

A GPU is a factory with a warehouse.

- **Compute units** are the workers: thousands of them, very fast.
- **Memory (HBM/GDDR)** is the warehouse: large (24 GB on the L4), but everything has to be trucked in over a road
  with a fixed capacity. That capacity is the **bandwidth**, in bytes per second.

The factory only runs at full speed if trucks deliver material fast enough. Some jobs need little material per unit
of work (a big matrix multiply reuses every number many times). Others need a truckload for every bit of work (reading
all of a model's weights to produce one token). The first kind is limited by the workers, the second by the road.
**The roofline** is the picture that tells you which one you are.

## 2. The math

| Symbol | Meaning | Unit |
|---|---|---|
| *P* | peak compute | FLOP/s |
| *B* | memory bandwidth | bytes/s |
| *F* | FLOPs a kernel performs | FLOPs |
| *Q* | bytes a kernel moves to or from memory | bytes |
| *I* = *F* / *Q* | arithmetic intensity | FLOPs/byte |

**Roofline:** attainable FLOP/s = min(*P*, *B* × *I*)

**Ridge point:** *I*\* = *P* / *B*. Kernels with *I* < *I*\* are memory-bound; kernels above it are compute-bound.

**Matmul** C[M, N] = A[M, K] · B[K, N] in 16-bit:
*F* = 2·M·N·K, and *Q* = 2·(M·K + K·N + M·N).
For decode, M is the batch size and the weight (K × N) dominates *Q*, so **I ≈ M**.

**Bandwidth from the datasheet:** per-pin data rate × bus width ÷ 8.
L4: 12.5 Gbit/s × 192 pins ÷ 8 = 300 GB/s.

**Why measured < datasheet:**
- DRAM has to refresh and to open and close rows.
- Switching between reads and writes costs time.
- Protocol overhead eats part of the bus.
- Clocks drop when the chip hits its power limit (72 W on the L4).

## 3. How the probe measures (see `src/fastserve/hw/`)

| Probe | What it does | Bytes or FLOPs counted |
|---|---|---|
| `bandwidth.copy_bandwidth` | `dst.copy_(src)` over sizes from 4 KiB to 1 GiB | read N + write N = 2N |
| `bandwidth.read_bandwidth` | Triton kernel summing blocks of floats | read N (+ one float per block) |
| `matmul.matmul_throughput` | BF16/FP16/FP8/INT8 matmuls, M ∈ {1…8192}, N = K ∈ {1024…8192}, **L2 flushed before each run** | 2·M·N·K |
| `overhead.launch_overhead` | 1,000 tiny kernels launched one by one, then replayed as one CUDA graph | time per kernel |
| `telemetry.sample_during` | NVML power and SM-clock sampling while idle (measured first), copying, and doing BF16 matmuls | watts, MHz |

Timing uses CUDA events after warm-up. Each point is the **median** of repeated runs.

## 4. Prediction (written before the first run)

Back-of-envelope, from the datasheet and experience with similar GPUs:

| Quantity | Prediction | Reasoning |
|---|---|---|
| Read bandwidth, ≥ 256 MiB | 250–275 GB/s (83–92% of 300) | GDDR6 typically delivers 85–90% of peak on streaming reads |
| Copy bandwidth, ≥ 256 MiB | 230–265 GB/s | Read/write turnaround costs a little more than pure reads |
| Transfers of 1–32 MiB (inside the 48 MiB L2) | *Above* the memory bandwidth, perhaps 1.5–3× | They are served by the on-chip L2 cache, not by memory |
| 4 KiB transfers | well under 10 GB/s | Latency-bound: a few µs per kernel regardless of size |
| BF16 matmul peak | 75–100 TFLOP/s (60–80% of 121) | Large GEMMs usually reach 70–80%. The 72 W power cap may pull clocks down under sustained tensor load. |
| FP8 matmul peak | 1.6–1.9× the measured BF16 peak | 2× on paper, minus scaling and power effects |
| INT8 matmul peak | Similar to FP8, possibly lower (low confidence) | cuBLASLt INT8 paths may be less tuned on Ada |
| BF16 matmul at M = 1, N = K = 4096 | ~0.2% of peak | Memory-bound: 32 MiB of weights at ~260 GB/s ≈ 0.13 ms for 33.5 MFLOP ≈ 0.26 TFLOP/s |
| BF16 ridge point | ~330 FLOPs/byte (datasheet 403) | ~85 TFLOP/s ÷ ~260 GB/s |
| Kernel launch overhead | eager 4–10 µs; CUDA graph 1–3 µs | Python + PyTorch dispatch dominate eager launches |
| Power | idle 15–25 W; copy 40–60 W; BF16 matmul ~70 W (the cap) | Tensor-core math is the most power-hungry workload |

## 5. Result

<!-- BEGIN GENERATED: hw_summary -->
*NVIDIA L4 · driver 580.95.05 · CUDA 13.0 · PyTorch 2.14.0+cu130 · Triton 3.8.0 · host CPU unknown · run `aed53c62b464` · commit `d5e40f5` · 2026-09-29T21:20:30+00:00*

| Quantity | Measured | Datasheet | Measured / datasheet |
|---|---|---|---|
| Memory bandwidth, read (GB/s) | 262 | 300 | 87% |
| Memory bandwidth, copy (GB/s) | 231 | 300 | 77% |
| BF16 matmul peak (TFLOP/s) | 57.0 | 121 | 47% |
| FP16 matmul peak (TFLOP/s) | 56.6 | 121 | 47% |
| FP8 matmul peak (TFLOP/s) | 117.8 | 242 | 49% |
| INT8 matmul peak (TOPS) | 128.1 | 242 | 53% |
| BF16 ridge point (FLOPs/byte) | 217 | 403 | — |
| FP8 ridge point (FLOPs/byte) | 449 | 807 | — |
| Kernel launch, eager (µs per kernel) | 8.72 | — | — |
| Kernel launch, CUDA graph (µs per kernel) | 0.95 | — | — |
| Power, idle (W) | 30 | — | — |
| Power, streaming memory (W) | 63 | — | — |
| Power, BF16 matmul (W) | 71 | 72 | 99% |
| SM clock, sustained BF16 matmul (MHz) | 1,009 | 2,040 | 49% |
| BF16 datasheet peak at that clock (TFLOP/s) | 59.8 | 121 | 49% |
| Temperature start → end (°C) | 61 → 68 | — | — |
<!-- END GENERATED: hw_summary -->

![Memory bandwidth vs transfer size](../../results/figures/hw_bandwidth_vs_size.png)
<!-- BEGIN GENERATED: caption-hw_bandwidth_vs_size -->
*Large reads reach 262 GB/s, 87% of the 300 GB/s datasheet figure; transfers that fit in L2 peak at 1365 GB/s (cache, not memory).*
<!-- END GENERATED: caption-hw_bandwidth_vs_size -->

![Roofline](../../results/figures/hw_roofline.png)
<!-- BEGIN GENERATED: caption-hw_roofline -->
*Measured BF16 ridge point: 217 FLOPs/byte (FP8: 449); an M=1 matmul (decode at batch 1) sits at 1.0 FLOPs/byte and reaches 0.2% of the measured peak.*
<!-- END GENERATED: caption-hw_roofline -->

![Matmul efficiency](../../results/figures/hw_matmul_efficiency.png)
<!-- BEGIN GENERATED: caption-hw_matmul_efficiency -->
*BF16 reaches 47% of datasheet peak at M=8192 but only 0.1% at M=1: decode-sized matmuls can't fill the GPU.*
<!-- END GENERATED: caption-hw_matmul_efficiency -->

### Prediction vs measurement

<!-- BEGIN GENERATED: m0_predictions -->
*Predictions written in commit `82f62cc`, before the first measurement.*

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| Read bandwidth, transfers >= 4x L2 (GB/s) | 250 – 275 | 262 | within range |
| Copy bandwidth, transfers >= 4x L2 (GB/s) | 230 – 265 | 231 | within range |
| Best 1-32 MiB transfer vs memory bandwidth (x) | 1.5 – 3 | 5.2 | above range |
| 4 KiB transfer (GB/s) | 0 – 10 | 0.348 | within range |
| BF16 matmul peak (TFLOP/s) | 75 – 100 | 57 | below range |
| FP8 peak / BF16 peak (x) | 1.6 – 1.9 | 2.07 | above range |
| BF16 M=1, N=K=4096 (% of datasheet peak) | ~0.2 | 0.144 | 0.72× the prediction |
| BF16 ridge point (FLOPs/byte) | ~330 | 217 | 0.66× the prediction |
| Kernel launch, eager (us per kernel) | 4 – 10 | 8.72 | within range |
| Kernel launch, CUDA graph (us per kernel) | 1 – 3 | 0.945 | below range |
| Power, idle (W) | 15 – 25 | 30 | above range |
| Power, streaming memory (W) | 40 – 60 | 63.4 | above range |
| Power, BF16 matmul (W) | ~70 | 70.9 | 1× the prediction |
<!-- END GENERATED: m0_predictions -->

**Explaining every gap:**

1. **Memory bandwidth landed in range.** Reads beat copies because a copy keeps switching the memory bus between
   reading and writing, and every switch costs time. Pure streaming reads avoid that.
2. **The L2 cache is much faster than I guessed** (above range). Copies of 16 MiB or less keep both source and
   destination inside the 48 MiB L2 from one iteration to the next, so they never touch memory. *Lesson: any
   benchmark that reuses small buffers measures the cache.* The first probe run fell into exactly this trap with
   matmuls (see "Dead end" below).
3. **BF16 peak below range: the L4 is power-limited.** This is the biggest lesson of M0.
   - Under sustained tensor-core math the GPU hits its 72 W cap and roughly **halves its SM clock** (row "SM clock,
     sustained BF16 matmul").
   - The datasheet's 121 TFLOP/s assumes the maximum clock. Scaled to the sustained clock (row "BF16 datasheet peak
     at that clock"), the measured peak is within a few percent of what the hardware can actually deliver.
   - cuBLAS isn't inefficient; the power budget is the ceiling.
4. **The ridge point is well below the datasheet's** (a direct consequence of 3). With a lower compute ceiling and
   near-datasheet bandwidth, the GPU turns compute-bound at a *smaller* arithmetic intensity, i.e. at a smaller
   batch size than the datasheet suggests. That matters for the batch-size crossover in M4.
5. **FP8 delivered its full 2× over BF16** (above my range). I expected extra losses from scaling and power, and
   they didn't show up. *Open question:* the clock wasn't sampled during FP8 matmuls, so the probe should add it.
6. **Batch-1 matmuls are slower than "weights ÷ bandwidth"** (below my point estimate). The prediction assumed an
   M=1 matmul streams its weight at full memory bandwidth. It doesn't. The smaller the weight, the further below it
   falls, because a small matmul can't keep enough memory requests in flight to fill the bus, and fixed start-up
   costs weigh more:

<!-- BEGIN GENERATED: m0_decode_matmuls -->
| Weight (N × K) | Size (MiB) | Time (µs) | Streamed at (GB/s) | vs measured read bandwidth |
|---|---|---|---|---|
| 1024 × 1024 | 2 | 18 | 114 | 43% |
| 2048 × 2048 | 8 | 56 | 149 | 57% |
| 4096 × 4096 | 32 | 193 | 174 | 66% |
| 8192 × 8192 | 128 | 663 | 203 | 77% |
<!-- END GENERATED: m0_decode_matmuls -->

   **Consequence for M1:** Qwen3-0.6B's weight matrices are small (a few MiB each; only the LM head is large).
   The batch-1 decode prediction must use these per-matmul streaming rates, not the best-case bandwidth, and it will
   come out well below the naive "model size ÷ bandwidth" ceiling.
7. **CUDA-graph replay beat my range; eager launches landed inside it.** Replaying a graph costs under a
   microsecond per kernel, roughly an order of magnitude less than launching eagerly from Python. That gap is the
   size of lever 3 on this setup.
8. **Idle and streaming power were above range.** "Idle" here means *a process holding a CUDA context*. The GPU
   stays in its top performance state with the SM clock near maximum even while doing nothing. Streaming memory also
   runs at maximum clock. Memory-bound work isn't cheap in watts on this GPU.

### Dead end (kept on purpose)

The first full probe run (`8088e303e4bc`, still in `results/raw/hw_probe.jsonl`) timed matmuls **with a warm
cache**. Its BF16 M=1, N=K=4096 result implied the 32 MiB weight was streaming at nearly **three times** the
measured memory bandwidth, which is impossible from memory. The weight simply stayed in the 48 MiB L2 between
iterations. The fix: `cuda_time_ms(..., flush_l2_bytes=...)` overwrites a scratch buffer twice the size of L2 before
every timed run, outside the timed region. A GPU test now asserts that a cold M=1 matmul can never stream faster
than the datasheet bandwidth. The same run also sampled "idle" power right after heavy work, so it read high; idle
is now measured first.

## 6. Check your understanding

1. The L4's BF16 ridge point is a few hundred FLOPs/byte. What does that mean for a matrix-vector product (decode at
   batch 1)?
   <details><summary>Answer</summary>A matrix-vector product does about 2 FLOPs per 2-byte weight, so its
   intensity is ≈ 1 FLOP/byte, hundreds of times left of the ridge. It is completely memory-bound. Its speed is set by
   bandwidth alone, and the tensor cores sit almost idle.</details>

2. Why is the measured bandwidth below the datasheet number?
   <details><summary>Answer</summary>The datasheet number is the raw pin rate. Real transfers lose time to DRAM
   refresh, opening and closing rows, read/write turnaround and protocol overhead, and clocks can drop at the power
   limit.</details>

3. Why can a 16 MiB transfer look *faster* than a 1 GiB one?
   <details><summary>Answer</summary>16 MiB fits in the L4's 48 MiB L2 cache, so repeated runs are served from
   on-chip cache, not memory. Only transfers much larger than L2 measure true memory bandwidth, which is why the probe's
   "measured bandwidth" uses sizes ≥ 4 × L2.</details>

4. What does the CUDA-graph number measure, and why is it lower than the eager one?
   <details><summary>Answer</summary>Eager launches pay Python + PyTorch dispatch + a CUDA launch per kernel. A CUDA
   graph records the whole sequence once and replays it with a single launch. What remains is roughly the GPU-side
   cost of running each tiny kernel back to back.</details>

5. FP8 doubles the peak FLOPs but leaves bandwidth unchanged. What happens to the ridge point, and to decode speed at
   batch 1 if only the *math* is FP8 while the weights stay BF16?
   <details><summary>Answer</summary>The ridge point doubles. Batch-1 decode is memory-bound, so faster math doesn't
   help at all. Only reading fewer bytes does, e.g. storing the weights themselves in FP8. That's lever 1, not a
   compute effect.</details>

## 7. Further reading

- S. Williams, A. Waterman, D. Patterson, *Roofline: An Insightful Visual Performance Model for Multicore
  Architectures* (2009).
- H. He, *Making Deep Learning Go Brrrr From First Principles* (blog post).
- NVIDIA L4 Tensor Core GPU datasheet.
- Triton tutorials: *Vector Addition* and *Fused Softmax*.
- PyTorch documentation: *CUDA Graphs*.
