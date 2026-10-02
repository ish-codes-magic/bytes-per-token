# Glossary

Terms as this project uses them, each with the place it is worked through. Definitions only: the numbers are
in the linked documents.

| Term | Meaning | Where |
|---|---|---|
| **Acceptance rate (α)** | Share of drafted tokens the target model keeps in speculative decoding. | [M6](learning/M6-speculative-decoding.md) |
| **Ablation** | Changing one thing at a time, everything else fixed, to attribute a result. A *ladder* adds techniques cumulatively; *leave-one-out* removes each from the full stack. | [M8](learning/M8-full-stack.md) |
| **Activation outlier** | A hidden channel whose values are far larger than the rest. It forces a quantization grid to waste its levels. | [M3](learning/M3-quant-reference.md) |
| **Arithmetic intensity** | FLOPs per byte moved. Low intensity means memory-bound, high means compute-bound. | [M0](learning/M0-foundations.md) |
| **Attention sink** | A token (usually the first) that many heads attend to strongly whatever the content. | [M1](learning/M1-nanoserve.md) |
| **AWQ** | Activation-aware weight quantization: scale salient channels up before rounding and compensate in the next layer, with the scales searched on calibration data. | [M3](learning/M3-quant-reference.md) |
| **Bandwidth (memory)** | Bytes per second the GPU can read from its own memory. The ceiling for memory-bound work. | [M0](learning/M0-foundations.md) |
| **BF16** | 16-bit floating point with a wide exponent; the stock serving precision here. | [M3](learning/M3-quant-reference.md) |
| **Calibration (model)** | Fitting a performance model's constants on measurements. Each constant here is fitted on a named group of earlier points, and the rest are held out. | [07](07-performance-model.md) |
| **Calibration (quantization)** | The sample text a quantizer uses to see realistic activations. | [M3](learning/M3-quant-reference.md) |
| **Closed loop / open loop** | Load generation with a fixed number of users who each wait for their answer, or with arrivals at a rate regardless of answers. They give different numbers for the same server. | [M2](learning/M2-baselines.md) |
| **Compute-bound** | Limited by FLOP/s: more arithmetic per byte than the ridge point. Prefill and large batches. | [M0](learning/M0-foundations.md) |
| **Continuous batching** | Adding and removing sequences from the running batch at every step instead of waiting for a batch to finish. | [M1](learning/M1-nanoserve.md) |
| **Control (experiment)** | A run that differs from another in exactly one suspected cause. It can only show an effect larger than whatever else bounds the measurement. | [M8](learning/M8-full-stack.md) |
| **CUDA graph** | A recorded sequence of kernel launches, replayed with one call. *Full*: the whole forward pass. *Piecewise*: the stretches between attention calls, with Python in between. | [06, section 6](06-results-analysis.md#6-the-pair-that-collides) |
| **Decode** | Generating tokens one at a time after the prompt. One new token per sequence per step: memory-bound at small batch. | [M1](learning/M1-nanoserve.md) |
| **Drafter** | The cheap proposer in speculative decoding: a small model, an n-gram lookup, or a head on the target's own features (EAGLE). | [M6](learning/M6-speculative-decoding.md) |
| **EAGLE-3** | A drafter that predicts from the target model's hidden features with one extra layer and a reduced vocabulary. | [M6](learning/M6-speculative-decoding.md) |
| **FlashAttention / FlashInfer** | Two attention kernel libraries vLLM can use. On this GPU an FP8 KV cache requires FlashInfer. | [M5](learning/M5-kv-cache.md), [06](06-results-analysis.md) |
| **FP8 (E4M3)** | 8-bit floating point: 4 exponent bits, 3 mantissa bits. Native tensor-core math on Ada and Hopper GPUs. | [M3](learning/M3-quant-reference.md) |
| **Goodput** | Requests per second that met the latency target. More honest than throughput, which counts late answers too. | [M2](learning/M2-baselines.md) |
| **GPTQ** | Quantizing a layer column by column while pushing each column's rounding error onto the columns not yet rounded, using the inverse Hessian of the layer's inputs. | [M3](learning/M3-quant-reference.md) |
| **Hadamard rotation** | Multiplying the residual stream by an orthogonal matrix that spreads outliers over all channels; the inverse is folded into the next weights, so the function is unchanged. | [M3](learning/M3-quant-reference.md) |
| **Host-bound** | A step that waits for the CPU-side program, not the GPU. Signs: GPU power below its limit, step time that does not follow model size. | [06, section 6](06-results-analysis.md#6-the-pair-that-collides) |
| **Interaction** | Combined speedup ÷ the product of the two single speedups. 1: they multiply. Below 1: they compete. | [M8](learning/M8-full-stack.md) |
| **KIVI layout** | KV-cache quantization with keys scaled per channel and values per token. | [M5](learning/M5-kv-cache.md) |
| **KL divergence** | Here: how far a modified model's next-token distribution is from the BF16 reference, per token, teacher-forced. The project's most sensitive quality measure. | [M2](learning/M2-baselines.md) |
| **KV cache** | The keys and values of every past token, kept so they are not recomputed. Grows with context and with batch. | [M1](learning/M1-nanoserve.md) |
| **Launch overhead** | The fixed cost of starting one GPU kernel from the host. It dominates tiny operations. | [M0](learning/M0-foundations.md) |
| **Memory-bound** | Limited by bandwidth: little arithmetic per byte. Decode at small batch. | [M0](learning/M0-foundations.md) |
| **nanoserve** | The project's minimal inference engine in plain PyTorch: the reference for every technique. | [M1](learning/M1-nanoserve.md) |
| **Needle in a haystack** | A recall test: a secret placed at some depth in a long prompt. Lossy KV techniques fail it first. | [M2](learning/M2-baselines.md) |
| **PagedAttention** | Storing the KV cache in fixed-size blocks with a block table, so sequences grow without copying. | [M1](learning/M1-nanoserve.md) |
| **Perplexity** | exp(mean negative log-likelihood) on held-out text. Coarser than KL. | [M2](learning/M2-baselines.md) |
| **Prefill** | Processing the prompt in one pass. Many tokens per weight read: compute-bound. | [M1](learning/M1-nanoserve.md) |
| **Prefix caching** | Reusing the KV blocks of a prompt prefix that was already processed. | [M5](learning/M5-kv-cache.md) |
| **Ridge point** | Peak FLOP/s ÷ bandwidth: the arithmetic intensity where the limit changes from memory to compute. | [M0](learning/M0-foundations.md) |
| **Roofline** | Attainable FLOP/s = min(peak, bandwidth × intensity). A bound on the GPU, not on the server. | [M0](learning/M0-foundations.md) |
| **RTN** | Round-to-nearest quantization: no calibration, no error compensation. | [M3](learning/M3-quant-reference.md) |
| **Speculative decoding** | Draft several tokens cheaply, verify them in one pass of the target, keep a prefix by a rejection rule that leaves the output distribution unchanged. | [M6](learning/M6-speculative-decoding.md) |
| **Split-KV** | Splitting one long sequence's attention across several GPU programs and merging exactly. | [M7](learning/M7-triton-kernels.md) |
| **TPOT / ITL** | Time per output token: the gap between a user's consecutive tokens. | [M2](learning/M2-baselines.md) |
| **TTFT** | Time to first token: queueing plus prefill plus one step. | [M2](learning/M2-baselines.md) |
| **Triton** | A Python-embedded language for GPU kernels: programs over blocks of data, compiled per shape. | [M7](learning/M7-triton-kernels.md) |
| **W4A16 / W8A8** | Weight and activation bit widths: 4-bit weights with 16-bit activations (dequantize, then multiply), or 8-bit both (multiply natively). | [M4](learning/M4-quant-production.md) |
