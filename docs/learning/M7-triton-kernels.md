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

*Filled in after the measurements.*
