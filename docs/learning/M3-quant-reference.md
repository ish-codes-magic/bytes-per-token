# M3: Quantization from scratch

M3 builds the main quantization methods ourselves, in plain PyTorch, and measures what each one costs in
quality on Qwen3-0.6B (and 1.7B): round-to-nearest, FP8 and NF4 formats, GPTQ, AWQ, Hadamard rotation, and W8A8
with SmoothQuant. Two library implementations (llm-compressor's GPTQ and AWQ) check that ours are right.

M3 is about **quality only**. Every method here is simulated ("fake quantization": round, then store the
rounded value back in BF16), which reproduces the exact error of low-bit storage but not its speed. Real low-bit
kernels, and what they buy, are M4.

## 1. Intuition

A weight is a number stored in 16 bits. Quantizing it means **snapping it to a coarse ruler**: 4 bits give 16
marks, and each group of weights gets its own ruler (a *scale*). The model then computes with the snapped values.

Three ideas carry the whole milestone:

1. **Every weight pays for its group's largest value.** The ruler has to reach the biggest weight in the group, so
   one outlier stretches the marks apart for everyone. Hence small groups (one ruler per 128 weights), and
   formats whose marks crowd where weights actually are (NF4, FP8).
2. **Only the output matters.** Rounding errors that cancel out in the layer's *output* are harmless. GPTQ uses
   this: after rounding one column of weights, it nudges the columns not yet rounded to compensate, guided by how
   the inputs correlate. AWQ uses it differently: it gives the weights that meet large inputs a finer ruler.
3. **Outliers can be moved, not just endured.** A few activation channels are 100–1000× larger than the rest.
   You can shift that difficulty into the weights (SmoothQuant), or rotate the whole space so no single channel
   stands out (Hadamard rotation). Neither changes what the model computes before rounding.

## 2. The math

### Grids

For b bits and a group of values x, with `qmax = 2^(b−1) − 1`:

```
symmetric     s = max|x| / qmax              q = clamp(round(x / s), −qmax − 1, qmax)      x̂ = s · q
asymmetric    s = (max − min) / (2^b − 1)    q = clamp(round(x / s) + z, 0, 2^b − 1)       x̂ = s · (q − z)
              z = round(−min / s)            (the integer that stands for a real 0)
```

- **Rounding error** is uniform in [−s/2, s/2], so its mean square is **s²/12**.
- **The textbook symmetric grid wastes a code.** INT4 has 16 codes (−8…7), but `s = max|x|/7` never uses −8.
  compressed-tensors (llm-compressor's format) uses `s = max|x| / 7.5` instead. Its step is 7% finer, and the
  largest positive value is clipped by half a step. We implement both (`full_range`).

**Granularity and storage.** Every group stores its scale in 16 bits (and, if asymmetric, a b-bit zero-point):

```
bits per weight = b + (16 + b·[asymmetric]) / (weights per scale)        INT4 g128 symmetric: 4.125
```

### Non-uniform formats

- **FP8** stores a sign, an exponent and a mantissa: `value = ±(1 + m/2^M) · 2^(e − bias)`. Each power of two
  holds the same number of points, so the *relative* error is about the same at any size. **E4M3** (M = 3, max
  448) is used for weights and activations; **E5M2** (M = 2, max 57,344) trades precision for range.
- **NF4** is a 16-value codebook at quantiles of a normal distribution (QLoRA). Weights are roughly bell-shaped,
  so each code covers about the same number of weights: every code is used equally often. (It isn't the
  codebook with the lowest squared error, which would be a Lloyd–Max quantizer, but it's close.)

### GPTQ, derived

**Objective.** Keep the layer's *outputs* on calibration inputs X [in, n] close, not the weights:

```
minimize ‖W X − Ŵ X‖²
```

Rows of W don't interact, so take one row w. Its error is a quadratic form in Δ = w − ŵ:

```
‖(w − ŵ) X‖² = Δ (X Xᵀ) Δᵀ = ½ · Δ H Δᵀ        with the Hessian  H = 2 X Xᵀ  [in, in]
```

H is the same for every row, and it only depends on the inputs. Its diagonal says how active each input is. Its
off-diagonals say which inputs move together.

**Rounding one weight, and compensating (Optimal Brain Surgeon).** Rounding weight q moves it by −e, where
e = w_q − ŵ_q. Let every weight not yet rounded (the set F, q included) move by δ, with δ_q = −e forced. The
smallest output error ½ δ H_F δᵀ under that one constraint (a Lagrange multiplier) is at

```
δ = − e / [H_F⁻¹]_qq · [H_F⁻¹]_q,F          (H_F: H restricted to F; check: δ_q = −e)
```

Weights whose inputs correlate with input q absorb most of its error. If H is diagonal (uncorrelated inputs),
[H⁻¹]_q,F is zero outside q and **GPTQ reduces to round-to-nearest**.

**Three tricks make this practical:**
1. **One order for every row.** Rounding columns left to right for all rows at once means every row uses the
   same sequence of H_F⁻¹: F shrinks by one column each step.
2. **Cholesky.** The rows [H_F⁻¹]_q,F for q = 1, 2, … are, up to a scale, exactly the rows of the upper Cholesky
   factor U of H⁻¹ (H⁻¹ = Uᵀ U). With d = U_qq, the update becomes `δ = −(e / d) · U_q,q:`. One factorization
   replaces an inverse per column.
3. **Lazy batches.** Updates inside a block of 128 columns are applied at once; the rest of W is updated once per
   block. Same arithmetic, far fewer passes over memory.

Plus **dampening** (add 1% of the mean diagonal to H so it's invertible), and **act-order** (process the most
active inputs first, while the most weights remain free to compensate). With act-order and groups, *static*
groups fix each group's grid from the original weights, so every group still maps to contiguous columns.

**Worked example** (4 inputs, 1 row, generated by `report/m3.py` from the actual `gptq_quantize`):

<!-- BEGIN GENERATED: m3_gptq_worked_example -->
<!-- END GENERATED: m3_gptq_worked_example -->

### AWQ

For any per-channel s > 0, `x Wᵀ = (x / s)(W · s)ᵀ` exactly. The division is folded into whatever produced x (an
RMSNorm's γ, or the previous linear's output rows), so it's free. Scaling up the weights that read large
activations gives them a finer effective grid:

```
s = x̄^α / w̄^(1−α)         x̄: mean |activation| per channel, w̄: mean |weight| per channel (relative to its group)
α* = argmin_α ‖ parent(Q(W · s) / s) − parent(W) ‖²      "parent": the attention block, the MLP, or the linear
```

### SmoothQuant

For W8A8 the *activations* are the problem, so move range from activations to weights with the same identity:

```
s_j = max|x_j|^α / max|W_j|^(1−α)          α = 0.5 splits each channel's difficulty evenly
```

### Rotation

For an orthogonal R (R Rᵀ = I): `x Wᵀ = (x R)(W R)ᵀ`. Rotating the residual stream mixes every channel into every
other, so a single huge channel becomes many moderate ones. It can be applied to the whole residual stream once
RMSNorm's per-channel γ is folded into the next linears, because `RMS(x R) = RMS(x)`. The LM head then needs its own
copy of the embedding (see `quant/rotation.py`).

## 3. Setup

- **Models:** Qwen3-0.6B and Qwen3-1.7B, BF16, in nanoserve (bit-exact with Hugging Face, M1).
- **What gets quantized:** the seven linear layers of every decoder layer. Embeddings and the LM head stay in BF16,
  as production recipes do. Qwen3 ties them, so quantizing the head would quantize the embedding too. For
  Qwen3-0.6B the embedding is a quarter of all weights, which caps how small the model can get.
- **Quality:** KL divergence from BF16 per token (teacher-forced), top-1 agreement and perplexity, on the same
  WikiText-2 test windows as M2 (`quality/perplexity.py`).
- **Calibration:** 128 sequences of 2,048 tokens of C4, the standard GPTQ/AWQ calibration set
  (`quality/text.py`). The calibration study swaps in WikiText (train), code, math and German.
- **Library check:** llm-compressor 0.14.0 quantizes the same model with the same calibration tokens. Its
  installed defaults, read from the source before writing the comparison:
  - GPTQ: act-order "static", block 128, dampening 0.01, and the full-range INT4 grid (`s = max|w| / 7.5`)
  - AWQ: duo scaling, a 20-point grid, the loss on each parent module's output, and no clipping

## 4. Prediction (written before quantizing a real model)

| Quantity | Prediction | Reasoning |
|---|---|---|
| BF16 perplexity in nanoserve | **19.3–19.8** | Same windows as M2's Hugging Face run, and nanoserve matches HF bit for bit (M1). Only attention kernels differ. |
| KL, RTN INT8 per-channel | **5e-5–2e-3** | Step = row max / 127: about 1% of a typical weight. Errors this small barely move the output distribution. |
| KL, RTN INT4 g128 | **0.05–0.25** | Step ≈ group max / 7: about 10% of a typical weight. Small models are fragile: published 4-bit perplexity increases for sub-1B models are 10–25%, i.e. ~0.1–0.2 nats. |
| INT4 per-channel / g128 | **1.5–10×** | A row spans 1,024+ inputs, and its max is set by the rare columns that read outlier channels. Groups of 128 contain the damage. |
| KL, RTN INT3 g128 | **0.25–2** | The step doubles, so squared error ×4, and 3-bit results in the literature degrade much faster than that. |
| KL, RTN INT2 g64 asymmetric | **2–12** | Four levels per group: the model stops being a language model. KL approaches the gap between its distribution and near-noise. |
| INT4 full-range / textbook grid | **0.8–1.0×** | A 7% finer step (squared error −13%) minus a little clipping at the positive extreme. |
| NF4 / INT4 at block 64 | **0.5–0.95×** | NF4 uses all 16 codes and places them where normal weights are dense. |
| KL, FP8 E4M3 weights | **5e-4–1e-2** | 3 mantissa bits: ~2% relative error per weight, about twice INT8 per-channel's, and far below INT4's. |
| GPTQ / RTN at INT4 g128 | **0.35–0.85×** | Error compensation on correlated inputs. For 4-bit g128 on larger models, GPTQ removes roughly half of RTN's perplexity increase. |
| AWQ / RTN at INT4 g128 | **0.4–0.9×** | Similar gains to GPTQ at 4 bits in the AWQ paper, from protecting the channels that meet large activations. |
| Our GPTQ / llm-compressor's | **0.85–1.15×** | Same algorithm, calibration tokens and grid. Differences: accumulation order and dtype details. |
| Our AWQ / llm-compressor's | **0.8–1.25×** | Same scale formula and loss, but our α search runs on a subset of the calibration sequences. |
| Rotated / plain RTN INT4 per-channel | **0.15–0.7×** | Rotation spreads outlier columns over every column, so row maxima stop being set by a few extreme weights. Only the residual-stream side is rotated here. |
| KL, W8A8 FP8 per-token | **2e-3–3e-2** | FP8 weights (above) plus FP8 activations at a similar relative error. |
| KL, W8A8 INT8 static per-tensor | **0.5–10** | One activation scale must cover channels 100–1000× larger than typical, so ordinary values round to a handful of levels. |
| SmoothQuant / plain INT8 static | **0.02–0.5×** | α = 0.5 takes the square root of each channel's range: a 1,000× outlier becomes ~30×. |
| Outlier ratio in the residual stream | **100–10,000×** | "Massive activations": a few channels in the residual stream reach thousands, while typical channels stay near 1. |
| down_proj's share of single-module INT4 KL | **25–60%** | Its input (after SwiGLU) has the largest outliers, and it writes straight into the residual stream. 7 module types → 14% if uniform. |
| First two + last layers' share | **15–60%** | Edge layers set up and read out the residual stream (and host the massive activations). 3 of 28 layers → 11% if uniform. |
| GPTQ with 8 / 128 calibration sequences | **1.0–1.4×** | H needs enough tokens to be well estimated. Past a few thousand tokens per 1,024-dim input, returns flatten. |
| GPTQ calibrated on code / on C4 | **1.0–1.4×** | Code has different activation statistics than the prose we evaluate on. GPTQ is known to be fairly robust to this. |
| GPTQ calibrated on WikiText / on C4 | **0.8–1.0×** | Calibrating on the evaluation's own domain can only help, a little. |
| Qwen3-1.7B / 0.6B, RTN INT4 g128 | **0.3–0.9×** | Bigger models have more redundancy and quantize more gracefully. |

## 5. Result

### Prediction vs measurement

<!-- BEGIN GENERATED: m3_predictions -->
<!-- END GENERATED: m3_predictions -->

### Grids and formats (no calibration)

<!-- BEGIN GENERATED: m3_grids -->
<!-- END GENERATED: m3_grids -->

### Calibrated methods, and the library check

<!-- BEGIN GENERATED: m3_calibrated -->
<!-- END GENERATED: m3_calibrated -->

### Weights and activations (W8A8)

<!-- BEGIN GENERATED: m3_w8a8 -->
<!-- END GENERATED: m3_w8a8 -->

### Explaining every gap

*(Written after the runs.)*

## 6. Check your understanding

1. Why does quantizing in groups of 128 make 4 bits viable, and what does it cost in bits?
   <details><summary>Answer</summary>Each group's scale only has to reach its own largest weight, so an outlier
   stretches the step for 127 neighbours instead of a whole row. Each group stores a 16-bit scale: 16 / 128 =
   0.125 extra bits per weight (more with a zero-point).</details>
2. GPTQ rounds each weight to the nearest grid point, like RTN. So where does its advantage come from?
   <details><summary>Answer</summary>From the weights it hasn't rounded yet. After rounding one column, it moves
   the remaining columns to cancel that column's error in the layer's output, weighted by how the inputs
   correlate (H⁻¹). Those moved weights are then rounded from their new values. With uncorrelated inputs there is
   nothing to cancel, and GPTQ equals RTN.</details>
3. Why does a Hadamard rotation remove outliers without changing the model's output?
   <details><summary>Answer</summary>It's orthogonal: x Wᵀ = (x R)(W R)ᵀ, so rotating inputs and weights
   together changes nothing. But x R mixes all channels, so one huge channel becomes many moderate ones, and
   grids fit better. RMSNorm's γ has to be folded into the next weights first, because only the pure RMS
   normalization commutes with a rotation.</details>
4. Why does INT8 per-tensor activation quantization fail on Qwen3 while FP8 per-tensor mostly survives?
   <details><summary>Answer</summary>INT8's 256 evenly spaced levels must span the largest outlier, so ordinary
   activations, hundreds of times smaller, land on a few levels near zero. FP8's levels are spaced
   logarithmically: small values still get fine steps, with about the same *relative* error as large
   ones.</details>
5. AWQ scales channels up; SmoothQuant scales them down. How can both be right?
   <details><summary>Answer</summary>They solve different problems with the same identity x Wᵀ = (x/s)(W·s)ᵀ. AWQ
   quantizes only weights: it scales up the weights that meet large activations, so their rounding error shrinks
   relative to their size. SmoothQuant quantizes activations too: it scales down the outlier activation
   channels so one scale can fit them, and the weights absorb the range.</details>

## 7. Further reading

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022.
- Hassibi & Stork, *Second order derivatives for network pruning: Optimal Brain Surgeon*, 1993.
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023.
- Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*, 2022.
- Dettmers et al., *QLoRA* (NF4), 2023; *LLM.int8()* (outlier features), 2022.
- Ashkboos et al., *QuaRot*, 2024; Liu et al., *SpinQuant*, 2024.
- Sun et al., *Massive Activations in Large Language Models*, 2024.
- Micikevicius et al., *FP8 Formats for Deep Learning*, 2022.
