---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
base_model_relation: quantized
library_name: vllm
tags:
- compressed-tensors
- llm-compressor
- fp8
- w8a8
- bytes-per-token
---

# Qwen3-1.7B, FP8 W8A8

A quantized [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), made with llm-compressor and served by vLLM. It is part of [bytes-per-token](https://github.com/ish-codes-magic/bytes-per-token), a project that measures where every speed and cost gain in LLM serving comes from.

## Use

```bash
vllm serve ishita-codes-ai/Qwen3-1.7B-FP8-Dynamic
```

## How it was made

- **Method:** FP8 (E4M3) weights with one scale per output channel, and FP8 activations quantized **per token at runtime** (llm-compressor's `FP8_DYNAMIC` scheme). No calibration data is needed.
- **Calibration data:** None.
- **Left in BF16:** the embedding and the LM head (tied in Qwen3), and all norms.
- `recipe.yaml` in this repo is llm-compressor's own record of the recipe.

## Quality, against the BF16 original

| Metric | BF16 | This checkpoint |
|---|---|---|
| KL divergence from BF16, WikiText-2 (lower is better) | 0.000 | 0.020 |
| Perplexity, WikiText-2, computed by vLLM | 15.55 | 15.56 |
| GSM8K 5-shot, flexible-extract | 69.0% | 67.4% |
| GSM8K 5-shot, strict-match | 69.7% | 67.1% |
| MMLU 5-shot (20 questions per subject) | 62.8% | 62.0% |
| HumanEval pass@1 | 40.2% | 36.6% |

## Speed on one NVIDIA L4, vLLM 0.30.0

| Measure | BF16 | This checkpoint |
|---|---|---|
| Time per output token, one user (ms) | 14.72 | 10.08 |
| Decode throughput, batch 1 (tokens/s) | 68 | 98 |
| Decode throughput, batch 256 (tokens/s) | 3,962 | 4,441 |
| Saturated server, 512 users (tokens/s) | 1,596 | 1,678 |
| Time to first token, 8k-token prompt (ms) | 692 | 566 |

## Caveats

- Task scores use base-model-style few-shot prompts without the chat template, identically for every format. They are for comparing formats of the same model, not for leaderboards.
- Speed was measured on one NVIDIA L4 (24 GB) with prefix caching off; other GPUs will differ.

Every number here is generated from `results/raw/m4_production.jsonl` in [bytes-per-token](https://github.com/ish-codes-magic/bytes-per-token) at commit `e09097751a`; the methodology is in its M4 learning doc.
