---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
base_model_relation: quantized
library_name: vllm
tags:
- compressed-tensors
- llm-compressor
- int4
- w4a16
- gptq
- bytes-per-token
---

# Qwen3-0.6B, INT4 W4A16 (GPTQ)

A quantized [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), made with llm-compressor and served by vLLM. It is part of [bytes-per-token](https://github.com/ish-codes-magic/bytes-per-token), a project that measures where every speed and cost gain in LLM serving comes from.

## Use

```bash
vllm serve ishita-codes-ai/Qwen3-0.6B-W4A16-GPTQ
```

## How it was made

- **Method:** GPTQ rounds the weights to INT4 in groups of 128 (one BF16 scale per group, symmetric), compensating each rounding error with the not-yet-rounded weights. Activations stay BF16 (`W4A16`).
- **Calibration data:** 128 sequences × 2,048 tokens of C4 (English web text, allenai/c4).
- **Left in BF16:** the embedding and the LM head (tied in Qwen3), and all norms.
- `recipe.yaml` in this repo is llm-compressor's own record of the recipe.

## Quality, against the BF16 original

| Metric | BF16 | This checkpoint |
|---|---|---|
| KL divergence from BF16, WikiText-2 (lower is better) | 0.000 | 0.312 |
| Perplexity, WikiText-2, computed by vLLM | 19.54 | 25.33 |
| GSM8K 5-shot, flexible-extract | 41.7% | 12.8% |
| GSM8K 5-shot, strict-match | 41.1% | 17.7% |
| MMLU 5-shot (20 questions per subject) | 49.6% | 43.2% |
| HumanEval pass@1 | 18.9% | 12.2% |

## Speed on one NVIDIA L4, vLLM 0.30.0

| Measure | BF16 | This checkpoint |
|---|---|---|
| Time per output token, one user (ms) | 5.78 | 3.41 |
| Decode throughput, batch 1 (tokens/s) | 170 | 281 |
| Decode throughput, batch 256 (tokens/s) | 5,738 | 5,777 |
| Saturated server, 512 users (tokens/s) | 1,834 | 1,818 |
| Time to first token, 8k-token prompt (ms) | 370 | 341 |

## Caveats

- Task scores use base-model-style few-shot prompts without the chat template, identically for every format. They are for comparing formats of the same model, not for leaderboards.
- Speed was measured on one NVIDIA L4 (24 GB) with prefix caching off; other GPUs will differ.
- This checkpoint often keeps writing after its final GSM8K answer (40% of problems, vs 1% for BF16). Metrics that read the *last* number in an answer (flexible-extract) under-count it; strict-match reads the marked answer.

Every number here is generated from `results/raw/m4_production.jsonl` in [bytes-per-token](https://github.com/ish-codes-magic/bytes-per-token) at commit `e09097751a`; the methodology is in its M4 learning doc.
