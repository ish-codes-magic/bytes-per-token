"""Fill the generated blocks in docs/ and README.md from results/raw/, so no number is typed by hand.

Stdlib only, so it runs on the laptop:

    uv run --only-group local python scripts/render_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # the laptop doesn't install the package, only its stdlib-only modules

from fastserve.report.render import render_file  # noqa: E402
from fastserve.report.tables import hw_summary  # noqa: E402
from fastserve.results import latest_run, read_jsonl  # noqa: E402


def build_blocks() -> dict[str, str]:
    blocks: dict[str, str] = {}
    probe = REPO / "results" / "raw" / "hw_probe.jsonl"
    if probe.exists():
        blocks["hw_summary"] = hw_summary(latest_run(read_jsonl(probe)))
    # Each figure's one-line takeaway, written by the figure code: <!-- BEGIN GENERATED: caption-<name> -->
    for caption in (REPO / "results" / "figures").glob("*.caption.txt"):
        name = caption.name.removesuffix(".caption.txt")
        blocks[f"caption-{name}"] = f"*{caption.read_text(encoding='utf-8').strip()}*"
    return blocks


def main() -> int:
    blocks = build_blocks()
    missing_any = False
    for path in sorted([REPO / "README.md", *REPO.glob("docs/**/*.md")]):
        changed, missing = render_file(path, blocks)
        if changed:
            print(f"updated {path.relative_to(REPO)}")
        for name in missing:
            missing_any = True
            print(f"no data yet for block '{name}' in {path.relative_to(REPO)}")
    return 1 if missing_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
