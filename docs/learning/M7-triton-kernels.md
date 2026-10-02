# M7: Custom Triton kernels

M4–M6 changed *what* the GPU is asked to do: fewer bytes, more tokens per byte. M7 changes *how* it is asked.
The third lever is **less waste between bytes and math**: launches that do little, and data that makes a
round trip to memory between two steps that could have shared one.

Two kernels, each chosen from a measurement:

1. **RMSNorm + INT8 activation quantization, fused.** vLLM's W8A8-INT8 path normalizes a hidden state in one
   kernel and quantizes it in another. vLLM 0.30 has a fused op for FP8 and none for INT8.
2. **Decode attention that reads a quantized KV cache directly.** M5 showed 4-bit keys and values keep
   quality with KIVI's layout, but only by simulation: the "INT4 cache" was BF16 numbers that happened to lie
   on a grid. No kernel in vLLM reads an INT4 or INT8 cache. This one does, and nanoserve gets a cache that
   really stores codes.

Both kernels have a plain-PyTorch reference written first (`src/fastserve/kernels/reference.py`) and are
tested against it on shapes chosen to break them.

## 1. Intuition

**The warehouse and the workbench.** A GPU's memory (HBM) is a warehouse down one road. M0 measured how many
gigabytes per second travel along it, and for memory-bound work that road is the only thing that matters. Each of the 58
streaming multiprocessors (SMs) is a workshop with a small workbench (registers and shared memory). Math at
the bench is nearly free next to a trip down the road.

A **kernel launch** is a job order sent to the workshops. A Triton kernel is the instruction sheet for one
job, a **program**: "you are job number `pid`; fetch *these* bytes, do *this*, store *that*." The launch says
how many jobs there are (the **grid**), and the GPU hands them out to workshops.

**Fusion.** Two steps written as two kernels: fetch the part, do step one, ship the part back to the
warehouse, fetch it again, do step two, ship it back. Fused: fetch once, do both steps at the bench, ship the
result. Same math, half the trips. Fusion only helps memory-bound work, and decode is memory-bound.

**Kernel 1 in this picture.** RMSNorm needs every element of a token's hidden vector (for the mean of
squares). Quantization needs every element of the *normalized* vector (for its largest value) and then every
element again (to round it). Three passes. If one job holds the whole vector at its bench, all three passes
run there and the warehouse sees one fetch and one delivery.

**Kernel 2 in this picture.** The KV cache is an archive stored compressed (4-bit codes). The unfused way to
read it is to decompress the entire archive onto the warehouse floor (a float tensor 4–8× larger), then let
attention fetch that. The fused way fetches the compressed pages and decodes at the bench. The kernel goes a
step further and *never decodes the pages at all*: it re-scales the question instead (section 2.2).

**Split-KV.** One request with a 32k-token context is one long archive per KV head: 8 jobs for 58 workshops.
Cut each archive into chunks, let every workshop summarize one chunk ("my chunk's answer, and how much weight
my chunk deserves"), and combine the summaries. The combination is exact (section 2.4).

**When none of this matters.** If a job moves a few kilobytes, the paperwork of the job order dominates. A
decode step at batch 1 is thousands of such tiny orders. There the only currency is the *number* of orders,
and fusing two kernels saves one order and nothing else.

## 2. The math

### 2.1 Bytes moved: fused vs unfused (kernel 1)

A token's hidden vector has d elements of 2 bytes (BF16). The INT8 output has d bytes plus one 4-byte scale.

| | reads | writes | total |
|---|---|---|---|
| `rms_norm` | 2d | 2d | 4d |
| `scaled_int8_quant` | 2d | d + 4 | 3d + 4 |
| **two ops** | | | **7d + 4** |
| **fused** | 2d | d + 4 | **3d + 4** |

So fusing moves 2.33× fewer bytes. That is the ceiling on the speedup when the data doesn't fit in the L2
cache. When it does fit, the "trips" are to the cache, and when there are only a few tokens, they don't
matter at all.

The computation per token x (symbols: w the norm's weight, ε its epsilon):

    y = x / sqrt(mean(x²) + ε) · w        scale = max|y| / 127        code = round(y / scale)

### 2.2 Folding the grids (kernel 2)

The cache stores, in KIVI's layout (M5):

- **keys**: one grid (scale s, zero-point z) per *channel* d, shared by a *group* of 32 consecutive tokens;
- **values**: one grid per *token*.

A stored code c stands for the number s·(c − z). The score of token t for query q is a dot product with the
dequantized key:

    score_t = Σ_d q_d · s_d · (c_td − z_d)
            = Σ_d (q_d · s_d) · c_td  −  Σ_d q_d · s_d · z_d
              └── scaled query q′ ──┘     └──── one bias b ────┘

s and z don't depend on t inside a group, so **the grid moves onto the query**: compute q′ = q ⊙ s and the
bias b once per group (2D multiplications), then every token's score is a plain dot product of q′ with its
raw codes. Dequantizing the keys instead would cost 32 × D multiplications and additions per group, and a
place to put the result.

The output is a weighted sum of dequantized values, with attention weights p_t:

    out = Σ_t p_t · s_t · (c_t − z_t)
        = Σ_t (p_t · s_t) · c_t  −  Σ_t p_t · s_t · z_t
          └ scaled weight ┘         └─ one number, the same for every channel ─┘

The value grid is per token, so **it moves onto the attention weight**.

4-bit codes are packed two per byte, channel d with channel d + D/2. One loaded byte then splits into two
half-vectors with a mask and a shift, and the kernel works on the two halves separately.

### 2.3 Online softmax

Attention needs softmax over all T tokens, but a program sees them in blocks. Keep three running numbers per
query: m (the largest score so far), l = Σ exp(score − m), and acc = Σ exp(score − m) · value. When a new
block raises the maximum to m′, everything accumulated so far was computed relative to m and must be scaled
by exp(m − m′):

    m′ = max(m, max of the block's scores)
    l   ← l · exp(m − m′)   + Σ_block exp(score − m′)
    acc ← acc · exp(m − m′) + Σ_block exp(score − m′) · value

At the end acc / l is the attention output, because the common factor exp(−m) cancels. Subtracting the
running maximum keeps every exponent ≤ 0, so nothing overflows. This is FlashAttention's trick.

### 2.4 Merging splits

Split s covers a set of tokens and returns its own normalized output out_s and lse_s = log Σ exp(score)
over its tokens (= m + log l). The attention over all tokens is

    out = Σ_t exp(score_t) · v_t / Σ_t exp(score_t)
        = Σ_s [exp(lse_s) / Σ_s′ exp(lse_s′)] · out_s

because exp(lse_s) · out_s is exactly split s's un-normalized sum. The bracket is a softmax over the splits'
lse values. No approximation is involved, which is why the same merge can join *different kinds* of parts:
the kernel's splits over 4-bit codes, and plain attention over the few newest tokens that are still in full
precision (section 3).

### 2.5 Which bound applies?

Arithmetic intensity (M0) = FLOPs ÷ bytes moved. The L4's ridge point for BF16 tensor-core math, peak
FLOP/s ÷ bandwidth, is a couple of hundred FLOPs per byte (M0).

- Kernel 1: about a dozen FLOPs per element against 3 bytes: ~4 FLOPs/byte.
- Kernel 2: per token and KV head, two query heads × (D multiply-adds for the score + D for the output) ≈
  1,000 FLOPs against 148 bytes (INT4) to 512 bytes (BF16): 2–7 FLOPs/byte.

Both sit far to the left of the ridge: **memory-bound**, even allowing that plain FP32 arithmetic is several
times slower than tensor cores. Their ceiling is bytes ÷ bandwidth. Two things can keep a real kernel below
that ceiling: too few programs to keep every SM busy (hence split-KV), and too many instructions per byte
(unpacking nibbles and converting integers to floats are instructions, not FLOPs).

For small inputs neither bound applies. The time is launches × cost per launch.

### 2.6 Triton in one page

    @triton.jit
    def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)                 # which job am I?
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)    # fetch a block into registers
        tl.store(out_ptr + pid, tl.sum(x, axis=0))                   # do the math, deliver

    kernel[(num_programs,)](x, out, n, BLOCK=1024, num_warps=4)      # the grid: how many jobs

- `tl.program_id` identifies the job; everything a program touches is computed from it.
- A `tl.constexpr` argument is fixed at compile time. Triton compiles one binary per combination, and block
  shapes must be powers of two.
- `mask` handles sizes that don't fill a block: masked elements are not read and not written.
- `num_warps` is how many groups of 32 threads run one program. More warps finish a big block faster but
  leave room for fewer programs per SM. It is tuned, not derived.
- Triton decides how a block's elements are spread over threads and when shared memory is needed. You write
  the math on whole blocks.

## 3. Setup, and the profile that chose the kernels

Everything runs on the L4. Microbenchmarks run in the serving image, next to the code they are compared
with (vLLM 0.30's own ops, FlashInfer 0.6). The real model runs in nanoserve, as in M1–M6.

- **Kernel 1** is timed two ways: launched from Python ("eager"), and replayed from a CUDA graph, which is
  how vLLM runs a decode step and which removes Python and the launcher from the picture.
- **Kernel 2** is timed on one layer of Qwen3-0.6B's attention (16 query heads, 8 KV heads, head dim 128),
  with the L2 cache flushed before every run: in a real step, 27 other layers and the weights pass through
  the cache between two visits to the same layer's keys.
- **In nanoserve**, `QuantizedKVCache` stores real codes. New tokens wait in a full-precision *tail* until
  their group of 32 is complete (a per-channel key grid needs the whole group), then become codes. A decode
  step runs kernel 2 over the codes and plain attention over the tail, and merges them (2.4). Prefill uses
  the unfused reference path: prefill is compute-bound and not what the cache's bytes affect.

### Profile 1: where nanoserve's decode step goes (before M7)

<!-- BEGIN GENERATED: m7_profile -->
| Batch × context | Step (ms) | Attention (ms) | Share of step | Kernels launched | KV cache read (GB) | GB/s through attention | Share of M0's bandwidth |
|---|---|---|---|---|---|---|---|
| 1 × 512 | 44.9 | 6.8 | 15% | 1,963 | 0.06 | 9 | 3% |
| 1 × 4,096 | 57.9 | 12.8 | 22% | 1,963 | 0.47 | 37 | 14% |
| 1 × 16,384 | 75.1 | 50.4 | 67% | 1,963 | 1.88 | 37 | 14% |
| 1 × 32,000 | 139.7 | 99.5 | 71% | 1,963 | 3.67 | 37 | 14% |
| 8 × 4,096 | 128.0 | 85.2 | 67% | 1,963 | 3.76 | 44 | 17% |
<!-- END GENERATED: m7_profile -->

Attention is a small part of the step at short context and most of it at long context. It gets through the
cache at a small fraction of the bandwidth M0 measured: nanoserve's path expands K and V to one copy per
query head before PyTorch's kernel reads them, so the cache makes several trips. That is the opening for a
fused kernel, before a single bit is saved.

The step at short context is far longer than its kernels' GPU time: the GPU waits for Python between
launches. At long context the kernels are long enough to hide that. Any kernel can only shrink the part of
the step that is GPU work.

### Profile 2: vLLM's norm and quantization ops

<!-- BEGIN GENERATED: m7_ops_profile -->
| Tokens × width | `rms_norm` (µs) | `scaled_int8_quant` (µs) | Both, in a CUDA graph (µs) | Both, from Python (µs) | vLLM's fused norm + FP8 (µs) | Both: GB/s moved |
|---|---|---|---|---|---|---|
| 1 × 1,024 | 1.7 | 1.6 | 3.2 | 55.3 | 2.6 | 2 |
| 16 × 1,024 | 1.7 | 1.6 | 3.2 | 56.8 | 2.6 | 36 |
| 256 × 1,024 | 2.1 | 2.0 | 4.1 | 56.8 | 10.1 | 448 |
| 4,096 × 1,024 | 15.8 | 20.9 | 31.3 | 59.9 | 136.3 | 939 |
| 32,768 × 1,024 | 571.7 | 401.9 | 987.7 | 940.5 | 1,242.6 | 238 |
| 1 × 2,048 | 1.8 | 1.6 | 3.3 | 58.4 | 2.9 | 4 |
| 16 × 2,048 | 1.8 | 1.7 | 3.3 | 58.4 | 2.9 | 69 |
| 256 × 2,048 | 2.7 | 2.5 | 5.0 | 58.4 | 11.1 | 739 |
| 4,096 × 2,048 | 22.1 | 21.8 | 45.6 | 64.0 | 158.8 | 1,288 |
| 32,768 × 2,048 | 1,139.0 | 853.6 | 1,997.1 | 1,994.2 | 1,637.6 | 235 |
<!-- END GENERATED: m7_ops_profile -->

- For a handful of tokens (decode) each op costs a fixed amount, and the pair costs twice that. Nothing
  about the data matters. From Python the same pair costs far more: that is launch overhead, which a CUDA
  graph removes.
- For 32,768 tokens the two ops run at close to the memory bandwidth. They are well-written kernels; the
  only thing left to save is trips.
- In between, the data sits in the L2 cache and the ops run several times faster than memory could feed
  them.
- vLLM's own fused RMSNorm + FP8 op is *slower* than its two separate INT8 ops at thousands of tokens. A
  fused kernel is not automatically a fast one.

### Profile 3: does this explain M4's INT8 gap?

M4 found W8A8-INT8 slower at batch 1 than its bytes predict, and attributed it to the separate quantization
op. With the op's cost now measured:

<!-- BEGIN GENERATED: m7_m4_gap -->
| Model | TPOT from bytes (ms) | Measured TPOT (ms) | Unexplained (ms) | Quantization launches per step | Their cost (ms) | Share of the gap | Fusable with a norm: share of TPOT |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 4.13 | 4.50 | 0.37 | 112 × 1.59 µs | 0.18 | 47% | 2.0% |
| Qwen3-1.7B | 9.35 | 9.97 | 0.62 | 112 × 1.64 µs | 0.18 | 29% | 0.9% |
<!-- END GENERATED: m7_m4_gap -->

The launches account for a minority of the gap. **M4's explanation was incomplete**: the separate op is
real, but most of the unexplained time is somewhere else (the INT8 matmul kernel itself is the next
suspect; not measured here). And only the two quantizations per layer that follow an RMSNorm can be fused
with it, so in vLLM's decode step kernel 1 can save a percent or two. It was always the warm-up kernel; the
profile says so in numbers.

## 4. Prediction (written after the profile, before timing either kernel)

Inputs: M0's read bandwidth 262 GB/s; the profile above.

| Quantity | Prediction | Reasoning |
|---|---|---|
| Kernel 1 vs vLLM's two ops, 1 token, CUDA graph | **1.4–2.1×** | One launch instead of two, each a fixed cost. |
| Kernel 1 vs two ops, 4,096 tokens (in L2) | **1.2–2.3×** | 2.33× fewer bytes, but vLLM's ops run from the cache at 4–5× memory bandwidth; a Triton program per row may not keep up. |
| Kernel 1 vs two ops, 32,768 tokens | **1.9–2.4×** | Memory-bound on both sides: the ratio of bytes moved, if the kernel also reaches ~240 GB/s. |
| Kernel 1's share of the bandwidth, 32,768 tokens | **0.75–1.0** | One read, one write, almost no math per byte. |
| Kernel 1 vs vLLM's fused FP8 op, 32,768 tokens | **1.5–2.0×** | Same bytes; vLLM's op only reaches about half the bandwidth in the profile. |
| Kernel 1 vs two ops, 1 token, from Python | **0.7–1.8×** | Triton's launcher prepares arguments in Python on every call; it may cost as much as two native calls. |
| Codes identical to the reference | **≥ 99.9%** | Same formula, same precision; only exact ties can differ. |
| Codes identical to vLLM's two ops | **90–99.5%** | vLLM's norm writes BF16 before quantizing; that rounding moves a value by up to a quarter of a code. |
| Kernel 2 on BF16 vs PyTorch attention, 1 × 32,768 | **2.7–6×** | The profile: PyTorch's path runs at ~14% of bandwidth. A fused kernel at 40–85% is 3–6× faster. |
| Kernel 2: INT4 vs BF16, 1 × 32,768 | **1.5–3.4×** | 3.46× fewer bytes, more instructions per byte (unpack, convert). |
| Kernel 2: INT8 vs BF16, 1 × 32,768 | **1.2–1.9×** | 1.86× fewer bytes. |
| Kernel 2 on BF16 vs FlashInfer, 1 × 32,768 | **0.5–1.1×** | A hand-tuned CUDA kernel should match or beat generic Triton on the same bytes. |
| Kernel 2 on INT4 vs FlashInfer on FP16 | **1.1–3.5×** | Fewer bytes against a better kernel. |
| Kernel 2 vs dequantize-then-attend, INT4 | **30–200×** | The reference makes half a dozen float32 passes over a tensor 7× the size of the codes. |
| Share of bandwidth, BF16 / INT4, 1 × 32,768 | **0.4–0.85 / 0.3–0.7** | Memory-bound in principle; Triton's reductions across threads and the integer unpacking keep it off the ceiling. |
| Kernel 2 on INT4 vs PyTorch, 1 × 512 | **0.5–1.3×** | Launch-bound: one Triton launch plus a small merge against PyTorch's few launches. No win expected. |
| Kernel vs reference, worst relative error | **< 0.001** | Float32 sums in a different order. |
| Best tokens per program, INT4, 1 × 32,768 | **512–2,048** | Enough programs for 58 SMs several times over, each still doing real work. |
| No splitting ÷ best split | **4–20×** | 8 programs can use 8 of 58 SMs. |
| nanoserve step, INT4 kernel vs before, 1 × 512 | **0.75–1.0×** | Attention is 15% of the step and launch-bound; the cache's bookkeeping adds launches. |
| nanoserve step, INT4 kernel vs before, 1 × 32,000 | **1.8–2.6×** | Attention is 71% of 140 ms; if it nearly vanishes, what remains is the ~45 ms of launches plus bookkeeping. |
| nanoserve step, INT4 kernel vs before, 8 × 4,096 | **1.7–2.8×** | Same reasoning: 67% of 128 ms is attention. |
| nanoserve step, INT4 read the unfused way, 1 × 32,000 | **0.08–0.5×** | Slower than BF16: it dequantizes the whole cache every step. Fewer bytes stored is not fewer bytes moved. |
| KV bytes per token, INT4 ÷ BF16, as allocated | **0.28–0.30** | M5's sizing: 4.625 bits per element instead of 16. |
| KL, BF16 cache read by kernel 2 | **< 0.002** | Only float32-vs-BF16 rounding in attention (M6: enough to flip a near-tie now and then). |
| KL, INT8 codes read by kernel 2 | **0.0003–0.003** | M5's simulation: 0.0014. |
| KL, INT4 codes read by kernel 2 | **0.008–0.04** | M5's simulation: 0.032 with *every* key quantized. Here the newest tokens are still in full precision, which can only help. |
| Needle recall, INT4 codes read by kernel 2 | **≥ 95%** | M5's simulation: 100%. |

## 5. Result

### Prediction vs measurement

<!-- BEGIN GENERATED: m7_prediction_score -->
**21 of 29 predictions in range.**
<!-- END GENERATED: m7_prediction_score -->

<!-- BEGIN GENERATED: m7_predictions -->
*Predictions written in commit `eb13c86`, before the first measurement.*

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| Kernel 1 vs vLLM's two ops, 1 token × 2,048, in a CUDA graph (x faster) | 1.4 – 2.1 | 1.88 | within range |
| Kernel 1 vs vLLM's two ops, 4,096 tokens × 2,048 (data resident in L2), in a CUDA graph (x faster) | 1.2 – 2.3 | 2.5 | above range |
| Kernel 1 vs vLLM's two ops, 32,768 tokens × 2,048, in a CUDA graph (x faster) | 1.9 – 2.4 | 2.34 | within range |
| Kernel 1's bytes moved ÷ time at 32,768 × 2,048, as a share of M0's read bandwidth | 0.75 – 1 | 0.899 | within range |
| Kernel 1 vs vLLM's fused RMSNorm + FP8 op, 32,768 tokens × 2,048, in a CUDA graph (x faster) | 1.5 – 2 | 1.92 | within range |
| Kernel 1 vs vLLM's two ops, 1 token × 2,048, launched from Python (x faster) | 0.7 – 1.8 | 0.95 | within range |
| Kernel 1's codes identical to the reference's (share, worst shape) | 0.999 – 1 | 1 | within range |
| Kernel 1's codes identical to vLLM's norm-then-quantize (share, worst shape) | 0.9 – 0.995 | 0.966 | within range |
| Kernel 2 vs nanoserve's PyTorch attention, same BF16 cache, 1 × 32,768 tokens (x faster) | 2.7 – 6 | 5.52 | within range |
| Kernel 2 on INT4 codes vs on BF16, 1 × 32,768 (x faster; bytes say 3.46) | 1.5 – 3.4 | 2.83 | within range |
| Kernel 2 on INT8 codes vs on BF16, 1 × 32,768 (x faster; bytes say 1.86) | 1.2 – 1.9 | 1.82 | within range |
| Kernel 2 on BF16 vs FlashInfer on FP16, 1 × 32,768 (x faster) | 0.5 – 1.1 | 0.929 | within range |
| Kernel 2 on INT4 vs FlashInfer on FP16, 1 × 32,768 (x faster) | 1.1 – 3.5 | 2.63 | within range |
| Kernel 2 vs dequantize-then-attend on the same INT4 codes, 1 × 32,768 (x faster) | 30 – 200 | 76.9 | within range |
| Kernel 2 on BF16: cache bytes ÷ time, as a share of M0's read bandwidth, 1 × 32,768 | 0.4 – 0.85 | 0.882 | above range |
| Kernel 2 on INT4: cache bytes ÷ time, as a share of M0's read bandwidth, 1 × 32,768 | 0.3 – 0.7 | 0.722 | above range |
| Kernel 2 on INT4 vs nanoserve's PyTorch attention, 1 × 512 tokens (x faster) | 0.5 – 1.3 | 1.31 | above range |
| Kernel 2 vs the reference on the same INT4 codes, worst relative error over shapes | 0 – 0.001 | 3.53e-06 | within range |
| Best tokens per program, INT4, 1 × 32,768 | 512 – 2,048 | 128 | below range |
| No splitting (8 programs) ÷ the best split, INT4, 1 × 32,768 (x slower) | 4 – 20 | 22.1 | above range |
| nanoserve decode step, INT4 codes + kernel 2 vs BF16 + PyTorch attention, 1 × 512 (x faster) | 0.75 – 1 | 0.73 | below range |
| nanoserve decode step, INT4 codes + kernel 2 vs BF16 + PyTorch attention, 1 × 32,000 (x faster) | 1.8 – 2.6 | 2.6 | above range |
| nanoserve decode step, INT4 codes + kernel 2 vs BF16 + PyTorch attention, 8 × 4,096 (x faster) | 1.7 – 2.8 | 2.29 | within range |
| nanoserve decode step, INT4 codes read the unfused way vs BF16 + PyTorch attention, 1 × 32,000 (x faster) | 0.08 – 0.5 | 0.316 | within range |
| KV bytes per token as allocated, INT4 cache ÷ BF16 cache | 0.28 – 0.3 | 0.289 | within range |
| KL vs the BF16 cache, BF16 cache read by kernel 2, token-by-token decode (nats) | 0 – 0.002 | 0.00113 | within range |
| KL vs the BF16 cache, INT8 codes read by kernel 2 (nats; M5's simulation: 0.0014) | 0.0003 – 0.003 | 0.00121 | within range |
| KL vs the BF16 cache, INT4 codes read by kernel 2 (nats; M5's simulation: 0.032) | 0.008 – 0.04 | 0.0115 | within range |
| Needle recall, INT4 codes read by kernel 2 (M5's simulation: 100%) | 0.95 – 1 | 1 | within range |
<!-- END GENERATED: m7_predictions -->

### Worked example: the bytes, for this model

![Memory traffic](../../results/figures/m7_traffic.svg)
<!-- BEGIN GENERATED: caption-m7_traffic -->
*Fusing moves 2.3× fewer bytes per token in kernel 1 (7,172 → 3,076) and 7.9× fewer per cached token in kernel 2 (1,172 → 148); a BF16 cache costs 512 bytes for the same token.*
<!-- END GENERATED: caption-m7_traffic -->

### Kernel 1: one launch instead of two, one trip instead of two

<!-- BEGIN GENERATED: m7_norm_quant -->
| Tokens × width | PyTorch (µs) | vLLM, two ops (µs) | vLLM fused FP8 (µs) | **Kernel 1 (µs)** | vs two ops | vs fused FP8 | Kernel 1: GB/s moved |
|---|---|---|---|---|---|---|---|
| 1 × 1,024 | 22.7 | 3.1 | 2.6 | 1.6 | 1.97× | 1.61× | 2 |
| 16 × 1,024 | 27.0 | 3.2 | 2.6 | 1.6 | 2.00× | 1.62× | 31 |
| 256 × 1,024 | 36.3 | 4.1 | 10.0 | 2.3 | 1.82× | 4.45× | 350 |
| 4,096 × 1,024 | 468.0 | 29.0 | 136.4 | 13.2 | 2.19× | 10.33× | 954 |
| 32,768 × 1,024 | 13,179.8 | 990.2 | 1,254.3 | 405.4 | 2.44× | 3.09× | 249 |
| 1 × 2,048 | 23.9 | 3.3 | 2.8 | 1.7 | 1.88× | 1.62× | 4 |
| 16 × 2,048 | 30.4 | 3.3 | 2.8 | 1.8 | 1.86× | 1.57× | 55 |
| 256 × 2,048 | 52.2 | 4.9 | 11.1 | 3.0 | 1.64× | 3.68× | 521 |
| 4,096 × 2,048 | 1,941.6 | 52.3 | 160.7 | 20.9 | 2.50× | 7.68× | 1,203 |
| 32,768 × 2,048 | 26,370.2 | 2,000.1 | 1,641.1 | 853.8 | 2.34× | 1.92× | 236 |
<!-- END GENERATED: m7_norm_quant -->

Three regimes, three reasons, all in one table:

- **A few tokens:** the time is launches. One instead of two.
- **Thousands of tokens:** the data lives in the L2 cache, and both sides run several times faster than
  memory could feed them. Fewer trips to the cache still win.
- **Tens of thousands of tokens:** memory-bound. The speedup is the ratio of bytes moved (section 2.1), and
  the kernel moves its bytes at nearly the bandwidth M0 measured.

Launched from Python instead of a CUDA graph, the small sizes show no gain at all:

<!-- BEGIN GENERATED: m7_norm_quant_eager -->
| Tokens × width | PyTorch (µs) | vLLM, two ops (µs) | vLLM fused FP8 (µs) | **Kernel 1 (µs)** | vs two ops | vs fused FP8 | Kernel 1: GB/s moved |
|---|---|---|---|---|---|---|---|
| 1 × 1,024 | 346.6 | 57.3 | 54.3 | 61.4 | 0.93× | 0.88× | 0 |
| 16 × 1,024 | 332.8 | 59.4 | 51.2 | 60.4 | 0.98× | 0.85× | 1 |
| 256 × 1,024 | 334.3 | 57.3 | 51.2 | 63.5 | 0.90× | 0.81× | 12 |
| 4,096 × 1,024 | 430.6 | 60.4 | 166.9 | 60.4 | 1.00× | 2.76× | 209 |
| 32,768 × 1,024 | 13,187.1 | 990.2 | 1,323.5 | 422.9 | 2.34× | 3.13× | 238 |
| 1 × 2,048 | 338.4 | 58.4 | 52.8 | 61.4 | 0.95× | 0.86× | 0 |
| 16 × 2,048 | 329.8 | 57.4 | 50.2 | 59.4 | 0.97× | 0.84× | 2 |
| 256 × 2,048 | 334.3 | 58.4 | 51.2 | 60.4 | 0.97× | 0.85× | 26 |
| 4,096 × 2,048 | 1,953.3 | 69.1 | 299.0 | 60.9 | 1.13× | 4.91× | 413 |
| 32,768 × 2,048 | 26,405.4 | 1,995.8 | 1,605.6 | 850.9 | 2.35× | 1.89× | 237 |
<!-- END GENERATED: m7_norm_quant_eager -->

Triton's launcher does its bookkeeping in Python on every call, which costs about as much as vLLM's two
native calls. A fused kernel for tiny inputs is only worth having inside a CUDA graph.

### Kernel 2: fewer bytes, read once

<!-- BEGIN GENERATED: m7_attention -->
| Batch × context | PyTorch attention, BF16 (µs) | FlashInfer, FP16 (µs) | Dequantize then attend, INT4 (µs) | **Kernel 2, BF16** (µs) | **Kernel 2, INT8** (µs) | **Kernel 2, INT4** (µs) | INT4 kernel vs PyTorch | INT4 kernel vs FlashInfer |
|---|---|---|---|---|---|---|---|---|
| 1 × 512 | 52 | 17 | 148 | 41 | 38 | 40 | 1.31× | 0.44× |
| 1 × 2,048 | 148 | 40 | 318 | 58 | 47 | 45 | 3.30× | 0.89× |
| 1 × 8,192 | 714 | 139 | 3,402 | 175 | 105 | 86 | 8.30× | 1.62× |
| 1 × 32,768 | 3,199 | 539 | 15,751 | 580 | 319 | 205 | 15.62× | 2.63× |
| 4 × 512 | 110 | — | 338 | 57 | 45 | 44 | 2.49× | — |
| 4 × 2,048 | 658 | — | 3,178 | 177 | 108 | 87 | 7.56× | — |
| 4 × 8,192 | 3,038 | — | 15,056 | 580 | 317 | 204 | 14.91× | — |
| 4 × 32,768 | 12,354 | — | 63,440 | 2,291 | 1,232 | 788 | 15.67× | — |
| 16 × 512 | 632 | — | 3,207 | 178 | 109 | 88 | 7.17× | — |
| 16 × 2,048 | 2,756 | — | 15,078 | 582 | 321 | 205 | 13.46× | — |
| 16 × 8,192 | 11,244 | — | 60,772 | 2,298 | 1,232 | 813 | 13.83× | — |
| 16 × 32,768 | 44,427 | — | — | 9,000 | 4,689 | 3,259 | 13.63× | — |
| 64 × 512 | 2,766 | — | 14,988 | 588 | 321 | 210 | 13.18× | — |
| 64 × 2,048 | 10,811 | — | 60,972 | 2,308 | 1,232 | 800 | 13.52× | — |
| 64 × 8,192 | 44,236 | — | — | 9,099 | 4,711 | 3,296 | 13.42× | — |
<!-- END GENERATED: m7_attention -->

![Speedup heatmaps](../../results/figures/m7_speedup.svg)
<!-- BEGIN GENERATED: caption-m7_speedup -->
*Kernel 2 on INT4 codes is 1.3–16× the speed of nanoserve's PyTorch attention (0 of 15 shapes slower), and 0.44–2.63× FlashInfer's on a full-precision cache: fewer bytes win at long context, launch overhead decides the short ones.*
<!-- END GENERATED: caption-m7_speedup -->

The four panels separate what the headline number mixes:

1. **INT4 kernel vs nanoserve's PyTorch attention**: large everywhere it matters.
2. **Fusion alone** (the same BF16 bytes): most of panel 1. nanoserve's old path copied K and V for every
   query head before attending.
3. **Bytes alone** (INT4 vs BF16 through the same kernel): real at long context, short of what the sizes
   promise, and nothing at short context.
4. **Against FlashInfer**, a production kernel on a full-precision cache: on the same bytes FlashInfer is a
   little faster than kernel 2. INT4 codes win once the cache is a few thousand tokens long, and lose
   below that, where the kernel's fixed cost dominates.

![Kernel roofline](../../results/figures/m7_roofline.svg)
<!-- BEGIN GENERATED: caption-m7_roofline -->
*At 1 × 32,768 tokens kernel 2 reads the BF16 cache at 88% of the measured bandwidth and the INT4 cache at 72%, where PyTorch's path manages 16%; kernel 1 moves its bytes at 90% once they no longer fit in the cache (above the line, the data is in L2).*
<!-- END GENERATED: caption-m7_roofline -->

### Why not 90% of the bandwidth for INT4?

<!-- BEGIN GENERATED: m7_timing_views -->
| Batch × context | INT4 cache (MiB) | PyTorch attention, BF16: cold · eager · graph (µs) | FlashInfer, FP16: cold · eager · graph (µs) | Kernel 2, BF16: cold · eager · graph (µs) | Kernel 2, INT4: cold · eager · graph (µs) |
|---|---|---|---|---|---|
| 1 × 512 | 0.6 | 52 · 145 · 39 | 17 · 51 · 11 | 41 · 213 · 27 | 40 · 219 · 27 |
| 1 × 2,048 | 2.3 | 148 · 162 · 129 | 40 · 50 · 13 | 58 · 212 · 30 | 45 · 207 · 29 |
| 1 × 8,192 | 9.2 | 714 · 739 · 777 | 139 · 53 · 23 | 175 · 205 · 65 | 86 · 202 · 65 |
| 1 × 32,768 | 37.0 | 3,199 · 3,337 · 3,399 | 539 · 541 · 538 | 580 · 594 · 590 | 205 · 223 · 201 |
<!-- END GENERATED: m7_timing_views -->

Three clocks on the same call: **cold** (the cache was evicted first), **eager** (called from Python, cache
warm), **graph** (replayed from a CUDA graph, cache warm).

- For BF16 at 32,768 tokens, cold and graph agree: the cache is far bigger than L2, so it comes from memory
  either way, at close to the bandwidth. **Memory-bound.**
- For INT4 at 32,768 tokens the whole cache *fits in L2*, and the warm replay is still no faster than cold.
  **Memory was never the limit: instructions were.** At 8,192 tokens, warm, BF16 and INT4 take the same
  time: with memory out of the picture, time follows tokens.
- At 512 tokens the eager column is several times the others: that is Python (Triton's launcher, then the
  merge's handful of small PyTorch ops), not GPU work.

So the INT4 kernel's ceiling is arithmetic per token (unpack two nibbles, convert to float, multiply and
add for two query heads), and Triton's generic code for it. The roofline said "memory-bound, far left of
the ridge", and for BF16 and INT8 it was right. For INT4 the kernel ran out of instructions before the GPU
ran out of bandwidth: FLOPs are not the only thing a kernel executes.

### Tuning

![Autotuning landscape](../../results/figures/m7_tuning.svg)
<!-- BEGIN GENERATED: caption-m7_tuning -->
*For INT4 at 1 × 32,768 the best cell is 128 tokens per program with 1 warp (2,048 programs): not splitting is 22× slower, and the wrong warp count at that split (8) 3.0×.*
<!-- END GENERATED: caption-m7_tuning -->

I predicted large programs would be best. The opposite held: about 128 tokens per program and a single
warp. A program waits on one block's load at a time, so many small programs keep more memory requests in
flight, and with one warp a program's sums never leave the warp.

### In the real model

<!-- BEGIN GENERATED: m7_nanoserve -->
| Batch × context | KV cache and reader | Step (ms) | vs before M7 | KV cache (MiB) | $ per 1M tokens | KL of the next token vs BF16 | Same top token |
|---|---|---|---|---|---|---|---|
| 1 × 512 | BF16 + PyTorch attention (before M7) | 31.2 | 1.00× | 56 | 6.93 | — | — |
| 1 × 512 | BF16 + kernel 2 | 43.7 | 0.71× | 56 | 9.71 | 0.0002 | 100% |
| 1 × 512 | INT8 codes + kernel 2 | 44.5 | 0.70× | 30 | 9.88 | 0.0006 | 100% |
| 1 × 512 | INT4 codes + kernel 2 | 42.7 | 0.73× | 16 | 9.49 | 0.0257 | 100% |
| 1 × 512 | INT4 codes, dequantize then attend | 58.1 | 0.54× | 16 | 12.91 | 0.0259 | 100% |
| 1 × 4,096 | BF16 + PyTorch attention (before M7) | 30.1 | 1.00× | 448 | 6.68 | — | — |
| 1 × 4,096 | BF16 + kernel 2 | 43.3 | 0.69× | 448 | 9.63 | 0.0004 | 100% |
| 1 × 4,096 | INT8 codes + kernel 2 | 43.5 | 0.69× | 242 | 9.68 | 0.0006 | 100% |
| 1 × 4,096 | INT4 codes + kernel 2 | 44.1 | 0.68× | 130 | 9.81 | 0.0155 | 100% |
| 1 × 4,096 | INT4 codes, dequantize then attend | 58.6 | 0.51× | 130 | 13.03 | 0.0153 | 100% |
| 1 × 16,384 | BF16 + PyTorch attention (before M7) | 74.5 | 1.00× | 1,792 | 16.56 | — | — |
| 1 × 16,384 | BF16 + kernel 2 | 54.6 | 1.37× | 1,792 | 12.13 | 0.0003 | 100% |
| 1 × 16,384 | INT8 codes + kernel 2 | 55.1 | 1.35× | 966 | 12.25 | 0.0004 | 100% |
| 1 × 16,384 | INT4 codes + kernel 2 | 53.5 | 1.39× | 518 | 11.88 | 0.0135 | 100% |
| 1 × 16,384 | INT4 codes, dequantize then attend | 219.2 | 0.34× | 518 | 48.71 | 0.0133 | 100% |
| 1 × 32,000 | BF16 + PyTorch attention (before M7) | 139.4 | 1.00× | 3,500 | 30.99 | — | — |
| 1 × 32,000 | BF16 + kernel 2 | 54.4 | 2.56× | 3,500 | 12.10 | 0.0004 | 100% |
| 1 × 32,000 | INT8 codes + kernel 2 | 53.4 | 2.61× | 1,887 | 11.87 | 0.0005 | 100% |
| 1 × 32,000 | INT4 codes + kernel 2 | 53.5 | 2.60× | 1,012 | 11.90 | 0.0152 | 100% |
| 1 × 32,000 | INT4 codes, dequantize then attend | 441.3 | 0.32× | 1,012 | 98.07 | 0.0163 | 100% |
| 8 × 4,096 | BF16 + PyTorch attention (before M7) | 127.7 | 1.00× | 3,584 | 3.55 | — | — |
| 8 × 4,096 | BF16 + kernel 2 | 54.6 | 2.34× | 3,584 | 1.52 | 0.0004 | 100% |
| 8 × 4,096 | INT8 codes + kernel 2 | 55.6 | 2.30× | 1,932 | 1.54 | 0.0006 | 100% |
| 8 × 4,096 | INT4 codes + kernel 2 | 55.6 | 2.29× | 1,036 | 1.55 | 0.0261 | 88% |
| 8 × 4,096 | INT4 codes, dequantize then attend | 440.8 | 0.29× | 1,036 | 12.24 | 0.0252 | 88% |
<!-- END GENERATED: m7_nanoserve -->

![One decode step, before and after](../../results/figures/m7_timeline.svg)
<!-- BEGIN GENERATED: caption-m7_timeline -->
*One decode step at 1 × 16,384 tokens: the GPU works 69 of 72 ms before and 12 of 73 ms after; what is left is 2,582 small kernels with gaps between them (Python between launches), which no attention kernel can shorten.*
<!-- END GENERATED: caption-m7_timeline -->

- At long context the step gets much shorter, and **the BF16 cache read by kernel 2 is as fast as INT4
  codes**. Once attention is one short kernel per layer, the step is a couple of thousand tiny launches
  with Python between them. Fewer bytes cannot shorten a wait for Python. In nanoserve INT4 buys memory
  (a cache 3.5× smaller, exactly M5's sizing), not time.
- At short context the new cache is *slower*: its bookkeeping adds launches to a step that was already
  launch-bound.
- Reading the codes the unfused way (dequantize, then attend) is the slowest of all, slower than BF16.
  **Storing fewer bytes is not moving fewer bytes.**

<!-- BEGIN GENERATED: m7_quality -->
| KV cache and reader | KL vs BF16 cache (nats) | Same top token | Perplexity | BF16 perplexity | M5's simulated KL | Needle recall | M5's simulated recall |
|---|---|---|---|---|---|---|---|
| BF16 + kernel 2 | 0.0011 | 98.3% | 29.29 | 29.28 | — | — | — |
| INT8 codes + kernel 2 | 0.0012 | 98.2% | 29.30 | 29.28 | 0.0014 | — | 100% |
| INT4 codes + kernel 2 | 0.011 | 94.4% | 29.55 | 29.28 | 0.032 | 100% | 100% |
<!-- END GENERATED: m7_quality -->

Speed is reported with quality: INT8 codes are indistinguishable from the yardstick's own noise (the first
row: the same BF16 cache, read by a kernel that sums in float32 instead of BF16), and INT4 stays close and
keeps every needle.

### Cost: waterfall v4

![Waterfall v4](../../results/figures/m7_waterfall.svg)
<!-- BEGIN GENERATED: caption-m7_waterfall -->
*Waterfall v4, nanoserve at 1 × 32,000 tokens: reading the same BF16 cache with kernel 2 changes cost by -61%, and INT4 codes by -62% (KL 0.011 from the BF16 cache): once attention is fused the step is bound by Python's launches, so INT4 buys a 3.5× smaller cache, not time. These bars are nanoserve's, not vLLM's.*
<!-- END GENERATED: caption-m7_waterfall -->

### What it would mean in vLLM (a projection)

<!-- BEGIN GENERATED: m7_projection -->
*Short-context step: 5.8 ms; 28 layers.*

| Attention kernel and cache | One layer (µs) | Projected vLLM step (ms) | Measured in vLLM (ms) | Projection error |
|---|---|---|---|---|
| FlashInfer, full-precision cache | 539 | 20.9 | 20.6 | 1.4% |
| FlashInfer, timed with the write flush | 691 | 25.1 | 20.6 | 22.0% |
| Kernel 2, BF16 cache | 580 | 22.0 | — | — |
| Kernel 2, INT8 codes | 319 | 14.7 | — | — |
| Kernel 2, INT4 codes | 205 | 11.5 | — | — |
| vLLM's FP8 cache (measured only) | — | — | 13.8 | — |
<!-- END GENERATED: m7_projection -->

The kernels are not in vLLM. The table adds one layer's measured attention time, for every layer, to vLLM's
short-context step. Its first row checks that sum against a step vLLM really ran; the kernel 2 rows are
what the same sum predicts. An INT4 cache would then be faster than vLLM's FP8 cache at this context, with
the quality above. That is a prediction to test, not a result.

### Explaining every gap

| Observation | Explanation (lever) |
|---|---|
| Kernel 1 kept up with vLLM's ops when the data was in L2 (above my range) | One read and one write of a row are as cheap for Triton as for hand-written CUDA; I had assumed a per-row program would lag at cache speed. |
| Kernel 2 reached more of the bandwidth than predicted | Two things. I underestimated how close generic Triton gets to hand-written CUDA on a plain streaming read. And the eviction method changed after the predictions were written: timed the original way (the write-flush column of the table in [04-kernels.md](../04-kernels.md)) both shares fall inside their ranges. |
| Best split far smaller than predicted, no-split penalty larger | Throughput here is memory requests in flight, not work per program. Many small programs, one warp each. |
| INT4 is short of the bandwidth, BF16 is not | Instruction-bound: the same arithmetic per token on a quarter of the bytes, plus unpacking. |
| Kernel 2 on INT4 beat PyTorch even at 512 tokens (above range) | The cold timing hides Python behind the eviction. On GPU time the kernel wins; called from Python it loses (the eager column). I predicted the eager behaviour and defined the cold measurement. |
| nanoserve at 512 tokens is slower with the new cache (below range) | Launch-bound step, more launches. *More waste between bytes and math*, of my own making. |
| nanoserve's speedup at 32k is all fusion, none from INT4 | After fusion the step is bound by Python between launches. Bytes only matter when the GPU is the bottleneck. |
| M4's INT8 gap is mostly not the separate quantization op | Its launches are measured now, and they are a minority of the gap. |
| Cold timings were slower than warm replays of data too big for L2 | The eviction itself: overwriting a scratch buffer left modified lines to write back. Evicting by reading fixed it, and the vLLM projection confirms which is right. |

### When each kernel is worth it

- **Kernel 1 (fused norm + INT8):** inside a CUDA graph, always a little; for long prefills, about the
  ratio of bytes. Launched from Python on small inputs, never. In vLLM's decode step the ceiling is a
  percent or two, because only half the quantizations follow a norm and each is one launch.
- **Kernel 2 (quantized KV attention):** when the cache is long enough to be the step's main read: a few
  thousand tokens and up. Below that, a full-precision cache and a kernel with a smaller fixed cost win.
- **INT4 vs INT8 codes:** INT8 is free in quality and memory-bound; INT4 costs a little quality, saves the
  most memory, and is limited by arithmetic. If capacity is the constraint (M5), INT4. If only time is,
  INT8 gets most of the gain.
- **In an engine that is launch-bound** (nanoserve): fuse first. Bytes come second.

## 6. Check your understanding

1. Kernel 1 moves 2.33× fewer bytes than the two ops. Why is its speedup close to 2× for one token, and
   close to the ratio of bytes for 32,768?
   <details><summary>Answer</summary>For one token no bound on bytes applies: the time is launches, and
   one launch replaces two (a little less than 2× because the fused kernel does slightly more per launch).
   For 32,768 tokens the data no longer fits in the L2 cache, both sides run at the memory bandwidth, and
   time is proportional to bytes moved: (7d + 4) / (3d + 4).</details>
2. Walk through what one program of kernel 2 does, and say where each piece of data lives.
   <details><summary>Answer</summary>It owns one (sequence, KV head, split). The two query heads sharing
   that KV head are loaded once and stay in registers. For each block of 32 tokens it loads the group's key
   scale and zero-point and folds them into the queries; loads the key codes from memory, splits the
   nibbles, and takes dot products (scores); updates the running max, weight sum and output (online
   softmax, in registers); loads the value codes and each token's value grid, folds the scale into the
   attention weights, and accumulates. At the end it writes one partial output and one log-sum per query
   head. Only codes and grids are read from memory; nothing the size of the cache is written.</details>
3. Why does the kernel never dequantize a key?
   <details><summary>Answer</summary>A key's grid (scale s, zero-point z per channel) is shared by 32
   tokens. score = Σ q·s·(c − z) = Σ (q·s)·c − Σ q·s·z, so the scale can be multiplied into the query once
   per group and the zero-point becomes one bias. The codes are then used as they are. Dequantizing would
   cost 32 × D multiply-adds per group instead of 2D, and somewhere to put the result.</details>
4. The INT4 kernel reaches a smaller share of the bandwidth than the BF16 one. How do you know it is not
   memory-bound, and what is it bound by?
   <details><summary>Answer</summary>At 32,768 tokens the INT4 cache fits in the L2 cache. Replayed warm
   from a CUDA graph (no memory traffic, no Python) it takes as long as it does cold, so memory was not the
   limit. And warm at 8,192 tokens BF16 and INT4 take the same time: time follows tokens. It is bound by
   instructions per token: unpacking, converting, and the multiply-adds, in Triton's generic code.</details>
5. In nanoserve the BF16 cache read by kernel 2 is as fast as INT4 codes. Why, and what would make the
   bytes matter again?
   <details><summary>Answer</summary>After fusion, attention is one short kernel per layer and the GPU is
   idle most of the step: the step is thousands of launches with Python between them. Fewer bytes shorten
   GPU work, which is no longer the bottleneck. Removing the launch overhead (CUDA graphs, as vLLM uses)
   would make the step GPU-bound again, and then the cache's bytes would set its length.</details>

## 7. Further reading

- Tillet, Kung, Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*,
  2019, and the Triton tutorials (the fused softmax and layer-norm kernels are kernel 1's ancestors).
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022 (the
  online softmax); Dao, *Flash-Decoding for long-context inference*, 2023 (split-KV for decode).
- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, 2024 (per-channel keys,
  per-token values, the full-precision residual).
- Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*, 2025.
- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for Multicore
  Architectures*, 2009.
- The vLLM source: `vllm/compilation/passes/fusion/rms_quant_fusion.py` (how a production engine fuses
  norm and quantization, and for which formats).
