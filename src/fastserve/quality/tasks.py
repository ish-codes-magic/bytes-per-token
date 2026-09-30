"""A small lm-evaluation-harness suite, run through vLLM.

Scores are for *comparing configurations of the same model* (BF16 vs quantized, etc.), not for leaderboards:
Qwen3 is prompted base-model style (few-shot, no chat template), identically for every configuration.
"""

from __future__ import annotations

import os
from typing import Any

# (task, few-shot examples, cap on examples per (sub)task, the metric we report)
DEFAULT_SUITE = [
    ("gsm8k", 5, None, "exact_match,flexible-extract"),
    ("mmlu", 5, 20, "acc,none"),  # 57 subjects × 20 questions: a representative slice, minutes not hours
    ("humaneval", 0, None, "pass@1,create_test"),
]


def run_suite(
    model_path: str, suite: list[tuple] | None = None, *, max_model_len: int = 4096
) -> dict[str, Any]:
    """Run each task and return {task: {"score": ..., "stderr": ..., "metric": ..., "n": ...}}."""
    import lm_eval

    # HumanEval executes the generated code, which is acceptable inside this throwaway container.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    model_args = {
        "pretrained": model_path,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "max_model_len": max_model_len,
    }
    scores: dict[str, Any] = {}
    for task, shots, limit, metric in suite or DEFAULT_SUITE:
        out = lm_eval.simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=[task],
            num_fewshot=shots,
            limit=limit,
            confirm_run_unsafe_code=True,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
        )
        result = out["results"][task]
        stderr_key = metric.replace(",", "_stderr,")
        scores[task] = {
            "metric": metric,
            "score": result.get(metric),
            "stderr": result.get(stderr_key),
            "shots": shots,
            "limit": limit,
            "n": sum(
                out.get("n-samples", {}).get(t, {}).get("effective", 0) for t in out.get("n-samples", {})
            ),
        }
    return scores
