"""Why a GSM8K answer failed: a wrong number, a model repeating itself, no final answer, or a right answer
buried under more text.

A task score says *how often* a quantized model fails, not *how*. Sorting each failure into a bucket tells
"it reasons slightly worse" (well-formed answers with the wrong number) apart from "it degenerates" (loops,
rambling past the answer format), which look identical in the score but mean different things.

The suite reports lm-eval's flexible-extract metric, which takes the *last* number in the answer. A model that
writes the right "#### 18" and then keeps talking is scored by whatever number it said last; strict-match,
which reads the number after "####", still counts it as right.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

BUCKETS = ("correct", "right, then kept talking", "wrong answer", "looping", "no final answer")
FINAL_ANSWER = re.compile(r"####\s*-?[\d.,$]+")


def is_looping(text: str, n: int = 8, times: int = 3) -> bool:
    """True when some run of `n` words appears `times` or more: the model is repeating itself.

    Eight words repeated three times almost never happens in a genuine step-by-step solution.
    """
    words = text.split()
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return bool(grams) and max(grams.values()) >= times


def talks_past_answer(text: str, words: int = 3) -> bool:
    """True when at least `words` words follow the first "#### <number>": the model didn't stop there.

    The few-shot examples end each answer at "#### <number>" and start a new "Question:", which lm-eval uses
    as a stop sequence, so a model following the pattern stops right after its answer.
    """
    found = FINAL_ANSWER.search(text)
    return bool(found) and len(text[found.end() :].split()) >= words


def classify(response: str, correct: bool, strict: bool = False) -> str:
    """One of BUCKETS, checked in order.

    `correct` is flexible-extract (the reported metric), `strict` is strict-match. The few-shot examples end
    every answer with "#### <number>", so a well-formed answer has that marker; one without it either looped
    or ran out of tokens before finishing.
    """
    if correct:
        return "correct"
    if strict:
        return "right, then kept talking"
    if is_looping(response):
        return "looping"
    if "####" in response:
        return "wrong answer"
    return "no final answer"


def summarize(samples: list[dict[str, Any]], examples: int = 2, chars: int = 600) -> dict[str, Any]:
    """Bucket counts from lm-eval's logged GSM8K samples (one entry per document per filter).

    `correct` uses the flexible-extract filter, the metric M2 and M4 report; strict-match is kept alongside.
    """
    docs: dict[int, dict[str, Any]] = {}
    for entry in samples:
        doc = docs.setdefault(entry["doc_id"], {"response": entry["resps"][0][0]})
        doc[entry["filter"]] = bool(entry["exact_match"])
    buckets: Counter[str] = Counter()
    shown: dict[str, list[str]] = {b: [] for b in BUCKETS if b != "correct"}
    words = talked = 0
    for doc_id in sorted(docs):
        doc = docs[doc_id]
        response = doc["response"]
        bucket = classify(response, doc.get("flexible-extract", False), doc.get("strict-match", False))
        buckets[bucket] += 1
        words += len(response.split())
        talked += talks_past_answer(response)
        if bucket in shown and len(shown[bucket]) < examples:
            shown[bucket].append(response[:chars])
    n = len(docs)
    return {
        "n": n,
        "accuracy": buckets["correct"] / n if n else None,
        "strict_accuracy": sum(d.get("strict-match", False) for d in docs.values()) / n if n else None,
        "buckets": {b: buckets[b] for b in BUCKETS},
        "talks_past_answer": talked / n if n else None,
        "mean_words": words / n if n else None,
        "examples": shown,
    }
