"""Needle in a haystack: hide one fact at a chosen depth of a long filler text, then ask for it.

The filler is generated from templates (seeded): copyright-free, endless, and it never contains the needle.
A cell (context length × depth) passes when the answer contains the needle's secret number.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Any

SUBJECTS = [
    "The river",
    "A quiet village",
    "The old library",
    "Morning traffic",
    "The mountain trail",
    "A small bakery",
]
VERBS = ["changes with", "depends on", "was shaped by", "reflects", "grew because of", "is known for"]
OBJECTS = [
    "the seasons",
    "local traditions",
    "careful planning",
    "the weather",
    "its visitors",
    "a long history",
]
DETAILS = [
    "Few people notice this.",
    "It has always been so.",
    "Records from the past agree.",
    "Nobody minds.",
]

QUESTION = "What is the special magic number mentioned in the document? Answer with just the number."


@dataclass(frozen=True)
class NeedleCase:
    context_tokens: int
    depth: float  # 0 = start of the context, 1 = the very end
    secret: str
    prompt: str  # the full chat-formatted prompt


def filler_sentences(rng: random.Random):
    while True:
        yield f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}. {rng.choice(DETAILS)}"


def build_case(tokenizer: Any, context_tokens: int, depth: float, secret: str, seed: int = 0) -> NeedleCase:
    """Filler of ~context_tokens tokens with the needle inserted at `depth`, wrapped in the chat template."""
    needle = f"The special magic number mentioned in this document is {secret}."
    rng = random.Random(seed)
    sentences, count = [], 0
    for sentence in filler_sentences(rng):
        count += len(tokenizer.encode(" " + sentence, add_special_tokens=False))
        if count > context_tokens:
            break
        sentences.append(sentence)
    sentences.insert(round(depth * len(sentences)), needle)
    document = " ".join(sentences)
    messages = [{"role": "user", "content": f"{document}\n\n{QUESTION}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: answer directly
    )
    return NeedleCase(context_tokens=context_tokens, depth=depth, secret=secret, prompt=prompt)


def grid(tokenizer: Any, lengths: list[int], depths: list[float], secrets: list[str]) -> list[NeedleCase]:
    return [
        build_case(tokenizer, n, d, s, seed=zlib.crc32(f"{n}/{d}/{s}".encode()))  # stable across runs
        for n in lengths
        for d in depths
        for s in secrets
    ]


def passed(case: NeedleCase, answer: str) -> bool:
    return case.secret in answer
