---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
base_model_relation: quantized
library_name: vllm
tags:
- compressed-tensors
- llm-compressor
- int8
- w8a8
- smoothquant
- gptq
- bytes-per-token
---

# Qwen3-1.7B, INT8 W8A8

A quantized [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), made with llm-compressor and served by vLLM. It is part of [bytes-per-token](https://github.com/ish-codes-magic/bytes-per-token), a project that measures where every speed and cost gain in LLM serving comes from.

## Use

```bash
vllm serve ishita-codes-ai/Qwen3-1.7B-W8A8-INT8
```

## How it was made

- **Method:** SmoothQuant (strength 0.8) moves activation outliers into the weights, then GPTQ rounds the weights to INT8 with one scale per output channel. Activations are quantized to INT8 **per token at runtime** (llm-compressor's `W8A8` scheme).
- **Calibration data:** 128 sequences × 2,048 tokens of C4 (English web text, allenai/c4).
- **Left in BF16:** the embedding and the LM head (tied in Qwen3), and all norms.
- `recipe.yaml` in this repo is llm-compressor's own record of the recipe.

## Quality, against the BF16 original

| Metric | BF16 | This checkpoint |
|---|---|---|
| KL divergence from BF16, WikiText-2 (lower is better) | 0.000 | 0.028 |
| Perplexity, WikiText-2, computed by vLLM | 15.55 | 15.31 |
| GSM8K 5-shot, flexible-extract | 69.0% | 67.4% |
| GSM8K 5-shot, strict-match | 69.7% | 68.2% |
| MMLU 5-shot (20 questions per subject) | 62.8% | 61.7% |
| HumanEval pass@1 | 40.2% | 38.4% |

## Speed on one NVIDIA L4, vLLM 0.30.0

| Measure | BF16 | This checkpoint |
|---|---|---|
| Time per output token, one user (ms) | 14.72 | 9.91 |
| Decode throughput, batch 1 (tokens/s) | 68 | 99 |
| Decode throughput, batch 256 (tokens/s) | 3,962 | 4,486 |
| Saturated server, 512 users (tokens/s) | 1,596 | 1,690 |
| Time to first token, 8k-token prompt (ms) | 692 | 551 |

## Caveats

- Task scores use base-model-style few-shot prompts without the chat template, identically for every format. They are for comparing formats of the same model, not for leaderboards.
- Speed was measured on one NVIDIA L4 (24 GB) with prefix caching off; other GPUs will differ.

Every number here is generated from `results/raw/m4_production.jsonl` in [bytes-per-token](https://github.com/ish-codes-magic/bytes-per-token) at commit `e09097751a`; the methodology is in its M4 learning doc.
