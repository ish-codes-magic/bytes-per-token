"""Real prompts by task, for experiments where the *text* matters.

M2–M5 used random token ids: speed there depends only on lengths. Speculative decoding is different: how
often a draft is accepted depends on what is being written. So M6 uses four kinds of real requests:

    chat        open-ended instructions (Dolly): free-form prose, hard to guess
    code        Python function stubs to complete (HumanEval): boilerplate and repeated identifiers
    math        grade-school word problems (GSM8K): short reasoning steps with arithmetic
    summarize   news articles to summarize (CNN/DailyMail): the answer reuses the prompt's words

Each is a public dataset read in its own order, so every run sees the same prompts. A prompt is the task's
instruction wrapped in the model's chat template with thinking off (Qwen3 then answers directly).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _chat(row: dict[str, Any]) -> str | None:
    return row["instruction"] if not row["context"] else None  # standalone instructions only


def _code(row: dict[str, Any]) -> str:
    return f"Complete this Python function. Reply with the whole function.\n\n```python\n{row['prompt']}```"


def _math(row: dict[str, Any]) -> str:
    return f"{row['question']}\nSolve it step by step."


def _summarize(row: dict[str, Any]) -> str:
    article = " ".join(row["article"].split()[:450])  # ~600 tokens: long enough to copy from
    return f"Summarize this article in three sentences.\n\n{article}"


@dataclass(frozen=True)
class Task:
    repo: str
    config: str | None
    split: str
    streaming: bool
    build: Callable[[dict[str, Any]], str | None]  # a dataset row → the user's message (None: skip the row)


TASKS = {
    "chat": Task("databricks/databricks-dolly-15k", None, "train", False, _chat),
    "code": Task("openai/openai_humaneval", None, "test", False, _code),
    "math": Task("openai/gsm8k", "main", "test", False, _math),
    "summarize": Task("abisee/cnn_dailymail", "3.0.0", "test", True, _summarize),
}


def task_messages(task: str, n: int) -> list[str]:
    """The first n user messages of a task, in the dataset's order."""
    from datasets import load_dataset

    spec = TASKS[task]
    messages: list[str] = []
    for row in load_dataset(spec.repo, spec.config, split=spec.split, streaming=spec.streaming):
        message = spec.build(row)
        if message:
            messages.append(message)
        if len(messages) == n:
            return messages
    raise ValueError(f"{task} has only {len(messages)} prompts, {n} requested")


def chat_prompt_ids(tokenizer: Any, message: str) -> list[int]:
    """One user message as the token ids the model sees: chat template, generation prompt, thinking off."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def task_prompts(tokenizer: Any, task: str, n: int) -> list[list[int]]:
    return [chat_prompt_ids(tokenizer, message) for message in task_messages(task, n)]
