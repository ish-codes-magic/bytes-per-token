"""M2 quality baselines: perplexity (Hugging Face), needle-in-a-haystack and task scores (vLLM).

Each task runs in its own container, so one vLLM instance never shares a GPU with another.
"""

from __future__ import annotations

from typing import Any


def perplexity_task(model_path: str, cfg: dict[str, Any]) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fastserve.quality.perplexity import evaluate, token_windows

    # huggingface_hub 1.x only accepts namespaced ids: "wikitext" now lives at "Salesforce/wikitext".
    text = "\n\n".join(load_dataset(cfg["repo"], cfg["dataset"], split="test")["text"])
    ids = AutoTokenizer.from_pretrained(model_path)(text, add_special_tokens=False).input_ids
    windows = token_windows(ids, cfg["window"], cfg["max_windows"])
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    result = evaluate(windows.cuda(), lambda x: model(x).logits, batch=1)  # full-vocab logits are large
    return {"dataset": cfg["dataset"], "window": cfg["window"], "windows": len(windows), **result}


def needle_task(model_path: str, cfg: dict[str, Any]) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    from fastserve.quality.needle import grid, passed

    llm = LLM(
        model=model_path,
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=0.85,
        seed=0,
        kv_cache_dtype=cfg.get("kv_cache_dtype", "auto"),  # M5: "fp8" stores the KV cache in FP8 E4M3
    )
    cases = grid(llm.get_tokenizer(), cfg["lengths"], cfg["depths"], cfg["secrets"])
    outputs = llm.generate([c.prompt for c in cases], SamplingParams(temperature=0.0, max_tokens=24))
    cells = [
        {
            "length": c.context_tokens,
            "depth": c.depth,
            "secret": c.secret,
            "passed": passed(c, o.outputs[0].text),
            "answer": o.outputs[0].text.strip()[:80],
        }
        for c, o in zip(cases, outputs, strict=True)
    ]
    return {"cells": cells, "pass_rate": sum(c["passed"] for c in cells) / len(cells)}


def tasks_task(model_path: str, cfg: dict[str, Any]) -> dict[str, Any]:
    from fastserve.quality.tasks import DEFAULT_SUITE, run_suite

    suite = [tuple(t) for t in cfg["suite"]] if cfg.get("suite") else DEFAULT_SUITE
    return {"scores": run_suite(model_path, suite, kv_cache_dtype=cfg.get("kv_cache_dtype"))}


TASKS = {"perplexity": perplexity_task, "needle": needle_task, "tasks": tasks_task}


def vllm_perplexity_task(model_path: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Perplexity on M2's WikiText-2 windows, computed by vLLM itself from prompt logprobs (its real kernels).

    Compared with nanoserve's perplexity of the same checkpoint's rounded weights, it shows whether the
    low-bit kernel computes what the checkpoint says, independently of how good the quantization is.
    """
    import math

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from fastserve.quality.perplexity import token_windows
    from fastserve.quality.text import wikitext_eval_ids

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    windows = token_windows(wikitext_eval_ids(tokenizer), cfg["window"], cfg["max_windows"])
    llm = LLM(
        model=model_path,
        max_model_len=cfg["window"] + 16,
        gpu_memory_utilization=0.85,
        seed=0,
        kv_cache_dtype=cfg.get("kv_cache_dtype", "auto"),
    )
    outputs = llm.generate(
        [{"prompt_token_ids": w.tolist()} for w in windows], SamplingParams(max_tokens=1, prompt_logprobs=0)
    )
    nll, positions = 0.0, 0
    for window, out in zip(windows, outputs, strict=True):
        for token, logprobs in zip(window.tolist()[1:], out.prompt_logprobs[1:], strict=True):
            nll -= logprobs[token].logprob  # the true next token's log-probability
            positions += 1
    return {
        "window": cfg["window"],
        "windows": len(windows),
        "positions": positions,
        "perplexity": math.exp(nll / positions),
    }
