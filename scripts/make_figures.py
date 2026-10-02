"""Draw the figures from results/raw/, on any machine. No GPU and no Modal account needed.

    uv run --only-group quick python scripts/make_figures.py                  # all -> results/figures/
    uv run --only-group quick python scripts/make_figures.py --milestone m8   # one milestone
    uv run --only-group quick python scripts/make_figures.py --out /tmp/figures --check

`--check` compares each freshly drawn figure's caption with the committed one in results/figures/ and fails
if any differs. Captions are computed from the data, so this is a test that the committed figures still show
what the committed raw records say.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no display needed

from fastserve.viz.render import MILESTONES, render_all, stale_captions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--milestone", choices=MILESTONES, action="append", help="default: every milestone")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "figures")
    parser.add_argument(
        "--check", action="store_true", help="fail if a caption differs from the committed one"
    )
    args = parser.parse_args()

    written = render_all(REPO, args.out, args.milestone)
    figures = sorted({path.name.split(".")[0] for path in written})
    print(f"drew {len(figures)} figures into {args.out}")
    if args.check:
        stale = stale_captions(written, REPO / "results" / "figures")
        for name in stale:
            print(f"caption differs from the committed one: {name}")
        if stale:
            return 1
        print("every caption matches the committed one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
