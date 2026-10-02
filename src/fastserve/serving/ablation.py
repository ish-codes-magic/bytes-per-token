"""The M8 ablation plan: which vLLM servers to run, written as techniques and combinations of them.

A technique is one letter (benchmarks/configs/m8_ablation.yaml, `techniques`). A server's label is the letters
that are switched on, in the config's order: "base" (none), "w", "wk", "wkps". Writing the plan this way
makes "one change at a time" checkable: two labels that differ by one letter differ by one technique.

`expand` turns the plan into the server list that `experiments/m2.run_serving` reads. Stdlib only: the laptop
expands the plan too, to know which containers to start.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

BASE = "base"
SIZE = {"Qwen/Qwen3-0.6B": "0.6b", "Qwen/Qwen3-1.7B": "1.7b"}


def letters(label: str) -> str:
    """The techniques a label switches on: "wk-r2" → "wk", "base" → ""."""
    core = label.split("-")[0]
    return "" if core == BASE else core


def label_of(on: str, order: str) -> str:
    """Canonical label for a set of letters, in the config's order: {"k", "w"} → "wk"; none → "base"."""
    return "".join(letter for letter in order if letter in on) or BASE


def factorial(factors: list[str]) -> list[str]:
    """Every subset of the factors, from none to all: the ladder, leave-one-out and every pair at once."""
    subsets = [combo for r in range(len(factors) + 1) for combo in combinations(factors, r)]
    return ["".join(combo) or BASE for combo in subsets]


def labels(plan: dict[str, Any]) -> list[str]:
    """One model's server labels: its factorial, listed combinations and controls, then the repeats (-r2)."""
    wanted = [*factorial(plan.get("factorial", [])), *plan.get("combos", []), *plan.get("controls", [])]
    wanted = list(dict.fromkeys(wanted))  # a combination listed twice runs once
    return wanted + [f"{label}-r2" for label in plan.get("repeats", [])]


def server(model: str, label: str, plan: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The server entry for one model and label: path, arguments, drafter, and the workloads it runs."""
    techniques = config["techniques"]
    on = letters(label)
    unknown = set(on) - set(techniques)
    if unknown:
        raise ValueError(f"label {label!r} uses undefined techniques {sorted(unknown)}")
    entry: dict[str, Any] = {"name": f"{SIZE[model]}-{label}", "label": label, "model": model}
    args = list(config.get("common_args", []))
    for letter, technique in techniques.items():
        if letter in on:
            args += technique.get("args", [])
            if "path" in technique:
                if "path" in entry:
                    raise ValueError(f"label {label!r} switches on two checkpoints")
                entry["path"] = technique["path"].format(model=model.split("/")[-1])
            if "speculative" in technique:
                entry["speculative"] = technique["speculative"][model]
        else:
            args += technique.get("off_args", [])
    entry["args"] = args
    everywhere = [w for w in config["loads"] if w not in config.get("only_for", {})]
    extra = [w for w, who in config.get("only_for", {}).items() if label in plan.get(who, [])]
    entry["workloads"] = everywhere + extra
    return entry


def expand(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every server of the plan, for every model, in a stable order."""
    return [
        server(model, label, plan, config) for model, plan in config["plan"].items() for label in labels(plan)
    ]


def differs_by_one(a: str, b: str) -> str | None:
    """The single technique that separates two labels, or None if they differ by more or less than one."""
    diff = set(letters(a)) ^ set(letters(b))
    return next(iter(diff)) if len(diff) == 1 else None
