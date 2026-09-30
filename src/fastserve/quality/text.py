"""Text as token ids, for calibration and evaluation. Every source is a public dataset read in a fixed order.

Calibration data decides what "typical inputs" GPTQ, AWQ and SmoothQuant optimize for, so M3 varies it:

    c4          web text: the standard GPTQ/AWQ calibration set
    wikitext    Wikipedia articles: the same kind of text as the evaluation set (its train split)
    code        Python files from GitHub
    math        grade-school math problems with worked answers
    german      German Wikipedia: a different language, same tokenizer

Large datasets are streamed: only the first documents are downloaded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Source:
    repo: str
    config: str | None
    split: str
    fields: tuple[str, ...]  # joined with a newline when there are several
    streaming: bool


SOURCES = {
    "wikitext_test": Source("Salesforce/wikitext", "wikitext-2-raw-v1", "test", ("text",), False),
    "wikitext": Source("Salesforce/wikitext", "wikitext-2-raw-v1", "train", ("text",), False),
    "c4": Source("allenai/c4", "en", "validation", ("text",), True),
    "code": Source("codeparrot/codeparrot-clean-valid", None, "train", ("content",), True),
    "math": Source("openai/gsm8k", "main", "train", ("question", "answer"), False),
    "german": Source("wikimedia/wikipedia", "20231101.de", "train", ("text",), True),
}


def documents(name: str) -> Iterator[str]:
    """The source's documents, in the dataset's own order."""
    from datasets import load_dataset

    src = SOURCES[name]
    rows = load_dataset(src.repo, src.config, split=src.split, streaming=src.streaming)
    for row in rows:
        text = "\n".join(row[f] for f in src.fields).strip()
        if text:
            yield text


def token_stream(tokenizer: Any, name: str, n_tokens: int) -> list[int]:
    """At least n_tokens token ids: documents joined by blank lines, tokenized as they come."""
    ids: list[int] = []
    for text in documents(name):
        ids += tokenizer("\n\n" + text if ids else text, add_special_tokens=False).input_ids
        if len(ids) >= n_tokens:
            return ids[:n_tokens]
    raise ValueError(f"{name} has only {len(ids)} tokens, {n_tokens} requested")


def calibration_ids(tokenizer: Any, name: str, samples: int, seq_len: int) -> Any:
    """[samples, seq_len] token ids from consecutive text of one source."""
    import torch

    return torch.tensor(token_stream(tokenizer, name, samples * seq_len)).view(samples, seq_len)


def wikitext_eval_ids(tokenizer: Any) -> list[int]:
    """WikiText-2 test as one stream, exactly as M2's perplexity baseline tokenized it."""
    from datasets import load_dataset

    src = SOURCES["wikitext_test"]
    text = "\n\n".join(load_dataset(src.repo, src.config, split=src.split)["text"])
    return tokenizer(text, add_special_tokens=False).input_ids
