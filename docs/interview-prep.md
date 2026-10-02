# Interview prep: the 20 questions

> **A draft for the owner to rewrite.** These answers are the agent's. Say each one out loud, then replace it
> with your own words. An answer you cannot reconstruct without reading it is not yours yet. The numbers come
> from the table below, which is generated from the raw results; learn those, not the prose.

## Numbers to have ready

<!-- BEGIN GENERATED: key_numbers -->
| Quantity | Value | Measured in |
|---|---|---|
| GPU memory bandwidth, read | 262 GB/s | M0 |
| Peak matmul rate, BF16 · FP8 | 57 · 118 TFLOP/s | M0 |
| Ridge point, BF16 | 217 FLOPs per byte | M0 |
| Qwen3-0.6B: weights in BF16 | 1.19 GB (596 M parameters) | M1 |
| Qwen3-0.6B: KV cache per token, BF16 | 112 KiB | M1 |
| Qwen3-0.6B: one-user ceiling, bandwidth ÷ weight bytes | 220 tokens/s | M1 |
| Qwen3-0.6B: stock vLLM, one user · 64 users | 168 · 3,495 tokens/s | M8 |
| Qwen3-1.7B: weights in BF16 | 3.44 GB (1,721 M parameters) | M1 |
| Qwen3-1.7B: KV cache per token, BF16 | 112 KiB | M1 |
| Qwen3-1.7B: one-user ceiling, bandwidth ÷ weight bytes | 76 tokens/s | M1 |
| Qwen3-1.7B: stock vLLM, one user · 64 users | 67 · 1,955 tokens/s | M8 |
| Qwen3-1.7B: best measured stack, latency | `wps`, 2.25× stock | M8 |
| Qwen3-1.7B: best measured stack, capacity | `wkps`, 2.28× stock | M8 |
| Qwen3-1.7B: the full stack, one user | `wkps`, 1.13× stock | M8 |
| Host time per pass on piecewise CUDA graphs | about 25 ms | M8 |
| Serving model, frozen: median error · without speculation | 6.4% · 3.5% | M8 |
<!-- END GENERATED: key_numbers -->

Letters: `w` FP8 weights, `k` FP8 KV cache, `p` prefix caching, `s` speculative decoding.

## The project in one breath

**1. What is this project, in two sentences?**
I took a stock vLLM deployment of a small open model on one cheap GPU and made it cheaper per token with
quantization, KV-cache engineering, speculative decoding and custom kernels, building each from scratch
before measuring the production version. Every gain is attributed to one of three levers and checked against
a performance model whose predictions were written down first.

**2. What are the three levers?**
Move fewer bytes per token (quantized weights and KV cache). Get more tokens per byte moved (batching, prefix
caching, speculation). Waste less between the bytes and the math (kernel fusion, CUDA graphs). If a result
does not fit one of them plus roofline reasoning, I keep digging.

**3. What is the headline result?**
The best configuration is up to nearly three times cheaper than stock, depending on the workload, at the
same quality class, and the best configuration is *not* everything switched on. The honest headline is the
second half.

## Physics

**4. Why is decoding memory-bound?**
One user's step multiplies each weight by one token's activations: about one operation per byte read. The
GPU's ridge point is hundreds of operations per byte. So the step is a read of the weights, and its ceiling
is bandwidth divided by the bytes read.

**5. Then why does batching help, and when does it stop?**
The weights are read once per step however many sequences share it, so tokens per byte rise with the batch.
It stops when the arithmetic catches up with the read (the ridge point), or when the KV cache, which grows
with batch and context, becomes what is read or what fills memory.

**6. Prefill against decode?**
Prefill processes the whole prompt in one pass: many tokens per weight read, so it is compute-bound. Decode
adds one token per pass: memory-bound at small batch. They need different optimizations and different
metrics (TTFT against TPOT).

**7. Why measure the GPU instead of using the datasheet?**
Because the datasheet's bandwidth and FLOP/s are not reachable, and every prediction is a fraction of a
ceiling. Measured ceilings turn "it should be about twice as fast" into a number that can be wrong.

## Quantization

**8. INT4 or FP8?**
It depends on load. INT4 reads the fewest bytes but is expanded to 16 bits before the multiply, so it wins
when memory-bound: few users. FP8 reads more bytes and multiplies natively at twice the peak rate, so it wins
when compute-bound: a full batch. The measured crossover came later than my first estimate, which I got
wrong and corrected in the M4 gate report.

**9. Explain GPTQ in three sentences.**
Quantize a layer one column of weights at a time. After rounding a column, push its error onto the columns
not yet rounded, in the direction that least changes the layer's output on calibration inputs, which the
inverse Hessian of those inputs gives. Processing in blocks and using a Cholesky factor makes it fast.

**10. Why does a rotation remove outliers without changing the model?**
An orthogonal matrix applied to the residual stream and its inverse folded into the next weights cancel
exactly. A Hadamard matrix mixes every channel into every other, so a few huge channels become many
ordinary ones. It needs the norms folded first, and that folding can move the unevenness into the weights,
which is why it hurt plain rounding here before it helped GPTQ.

## KV cache

**11. Why are keys harder to quantize than values?**
Keys have a few channels much larger than the rest, the same in every token. One scale per token wastes the
grid on those channels. A scale per channel for keys and per token for values (KIVI's layout) fixes it.

**12. What does prefix caching save?**
The prefill of the shared part: its FLOPs and its share of TTFT. It saves nothing where prompts do not
repeat, and costs nothing measurable there either. In decode there is a second, smaller saving: shared blocks
are read once per group if a layer's slice of them fits in the GPU's cache.

## Speculative decoding

**13. Why is it lossless?**
Accept a drafted token with probability min(1, p/q), and on rejection sample from the normalized positive
part of p − q. Summing the two ways a token can come out gives exactly p. I tested it statistically, and the
test first failed for a reason that was not the algorithm: 16-bit arithmetic made two "identical" passes
differ, so the comparison has to be against each sampler's own pass.

**14. Why does it stop helping on a busy server?**
It spends idle compute on guesses. One user leaves the GPU's arithmetic almost unused, so verifying four
tokens costs about one read. At a full batch there is no idle compute, verification costs real time, and
rejected drafts are pure loss.

## Kernels

**15. Walk through your decode-attention kernel.**
Each program handles one KV head and a chunk of one sequence's cache. It loads 4-bit codes, and instead of
dequantizing them it folds the keys' scales into the query once and the values' scales into the attention
weights, so the inner loop is integer codes times a float. Chunks keep a running maximum and sum (online
softmax) and are merged exactly afterwards. It beats a full-precision library kernel only when the cache is
long, because then bytes dominate; at short context the launch cost decides.

**16. Why did it reach about three quarters of the bandwidth and not more on 4-bit codes?**
Because that path is no longer memory-bound. Replayed warm with its whole cache already in L2 it is no
faster than cold: the limit is instructions per token (unpacking and scaling), not bytes.

## The full stack

**17. Why is the full stack not the best stack?**
An FP8 KV cache forces a different attention library on this GPU, and with that library vLLM cannot record a
speculative pass as one CUDA graph. It falls back to piecewise graphs, Python runs between every layer, and
a step takes the longer of what the GPU and the host need. At low load the host needs more. Neither
technique is at fault; the combination changes the code path. On a newer GPU generation it should not happen.

**18. How did you find that, and what did you get wrong?**
Controls with predictions first. I forced the graph mode alone on one server, saw no loss, and cleared it.
That was wrong: the server's GPU work per pass was longer than the host's, so the host was hidden. I then
blamed the attention library's planning, timed it alone, and it was a small fraction of a pass. A profile
showed identical GPU work with the GPU idle half the time, and the same control on the smaller model showed
the loss at once. The lesson: a null result is only as good as what the measurement could have seen.

**19. How good is the performance model, honestly?**
Within a few percent wherever a step waits for the GPU, on configurations it was not fitted to. Wrong by a
large factor where a step waits for the host, because a model of bytes and FLOPs has no host. Its median
error looked fine while it missed the headline; a median hides a cluster.

## Judgment

**20. What would you do differently, or next?**
Log the engine's graph mode from the first run; I had to start servers again just to read it. Check the
billing between investigation rounds, not after. Next: find why FP8 weights double the host's time (one
profile, kept runnable), repeat the colliding servers on a Hopper GPU, and put the 4-bit attention kernel
behind vLLM's backend interface so its bar in the waterfall is a measurement instead of a projection.
