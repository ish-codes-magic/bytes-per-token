"""Compare two profiled servers call by call: where does one spend host time that the other does not?

    uv run --only-group local python scripts/compare_profiles.py 0.6b-wg 0.6b-g

Reads the `m8_profile` records in results/raw/m8_ablation.jsonl (written by `modal run ...::m8 --profile`)
and prints, per engine step, the host calls whose self time differs most between the two servers. This is
the tool for the question M8 left open (docs/open-questions.md).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fastserve.report import m8  # noqa: E402
from fastserve.report.tables import markdown_table  # noqa: E402
from fastserve.results import read_jsonl  # noqa: E402

SKIP = {"Trace", "gpu_user_annotation", "overhead"}  # the profiler's own rows, and the GPU's timeline
ADDRESS = re.compile(
    r" (?:object )?at 0x[0-9a-f]+"
)  # "<built-in method acquire of _thread.lock object at 0x..>"


def differences(first: dict, second: dict, top: int = 25) -> list[dict]:
    """Host calls ranked by |self time per step in `first` − in `second`|. A call missing from a profile's
    kept rows counts as zero there, and is marked. Object addresses are dropped from names, so the same
    call in two processes is one row."""
    per_step: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for key, profile in (("first", first), ("second", second)):
        for row in profile["host"]:
            if row["cat"] in SKIP:
                continue
            entry = per_step.setdefault((row["cat"], ADDRESS.sub("", row["name"])), {})
            sums = entry.setdefault(key, {"self_ms": 0.0, "calls": 0.0})
            sums["self_ms"] += row["self_ms"] / profile["iterations"]
            sums["calls"] += row["calls"] / profile["iterations"]
    rows = []
    for (cat, name), entry in per_step.items():
        a, b = entry.get("first"), entry.get("second")
        rows.append({"cat": cat, "name": name, "first": a, "second": b})
        rows[-1]["diff_ms"] = (a["self_ms"] if a else 0.0) - (b["self_ms"] if b else 0.0)
    return sorted(rows, key=lambda row: -abs(row["diff_ms"]))[:top]


def table(first: dict, second: dict, top: int = 25) -> str:
    def cell(entry: dict | None) -> str:
        return f"{entry['self_ms']:.2f} ms × {entry['calls']:.1f}" if entry else "not among the rows kept"

    rows = [
        [
            f"{row['diff_ms']:+.2f}",
            cell(row["first"]),
            cell(row["second"]),
            row["cat"],
            f"`{row['name'][:80]}`",
        ]
        for row in differences(first, second, top)
    ]
    headers = [
        "Difference (ms per step)",
        f"`{first['label']}`: self time × calls per step",
        f"`{second['label']}`",
        "Kind",
        "Call",
    ]
    return markdown_table(headers, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", help="a profiled server's name, e.g. 1.7b-wg")
    parser.add_argument("second", help="the server to compare it with, e.g. 1.7b-g")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    records = read_jsonl(REPO / "results" / "raw" / "m8_ablation.jsonl")
    profiles = [m8.profile_of(records, name) for name in (args.first, args.second)]
    for name, profile in zip((args.first, args.second), profiles, strict=True):
        if profile is None:
            print(f"no profile of {name} in results/raw/m8_ablation.jsonl: run ::m8 --profile --only {name}")
            return 1
        steps = profile["iterations"]
        print(
            f"{name}: {profile['span_ms'] / steps:.1f} ms per step under the profiler, "
            f"{profile['gpu_ms'] / steps:.1f} ms of GPU kernels"
        )
    print()
    print(table(*profiles, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
