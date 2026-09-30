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

    llm = LLM(model=model_path, max_model_len=cfg["max_model_len"], gpu_memory_utilization=0.85, seed=0)
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
    return {"scores": run_suite(model_path, suite)}


TASKS = {"perplexity": perplexity_task, "needle": needle_task, "tasks": tasks_task}
