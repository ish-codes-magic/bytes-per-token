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
| `matmul.matmul_throughput` | BF16/FP16/FP8/INT8 matmuls, M ∈ {1…8192}, N = K ∈ {1024…8192} | 2·M·N·K |
| `overhead.launch_overhead` | 1,000 tiny kernels launched one by one, then replayed as one CUDA graph | time per kernel |
| `telemetry.average_power_during` | NVML power sampling while idle, copying, and doing BF16 matmuls | watts |

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
<!-- END GENERATED: hw_summary -->

![Memory bandwidth vs transfer size](../../results/figures/hw_bandwidth_vs_size.png)
<!-- BEGIN GENERATED: caption-hw_bandwidth_vs_size -->
<!-- END GENERATED: caption-hw_bandwidth_vs_size -->

![Roofline](../../results/figures/hw_roofline.png)
<!-- BEGIN GENERATED: caption-hw_roofline -->
<!-- END GENERATED: caption-hw_roofline -->

![Matmul efficiency](../../results/figures/hw_matmul_efficiency.png)
<!-- BEGIN GENERATED: caption-hw_matmul_efficiency -->
<!-- END GENERATED: caption-hw_matmul_efficiency -->

**Prediction vs measurement:** *(written after the first run)*

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
