"""Build the dashboard's data from results/raw/: site/data/dashboard.json and site/highlight.html.

    uv run --only-group local python scripts/build_site.py          # write
    uv run --only-group local python scripts/build_site.py --check  # fail if the committed files are stale

The page itself (site/index.html, app.js, model.js, style.css) is static and hand-written. Serve it with any
static server, e.g. `python -m http.server -d site`, or open the GitHub Pages deployment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import render_docs  # noqa: E402
from fastserve.report.site import dashboard  # noqa: E402

DATA = REPO / "site" / "data" / "dashboard.json"
HIGHLIGHT = (REPO / "results" / "figures" / "m6_highlight.html", REPO / "site" / "highlight.html")


def build() -> str:
    """The data file's text: deterministic, so a stale committed copy can be detected."""
    data = dashboard(REPO, render_docs.build_blocks())
    # Keys keep their order: the page reads models, workloads and techniques in the order given here.
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed files are stale")
    args = parser.parse_args()
    text = build()
    source, target = HIGHLIGHT
    if args.check:
        stale = not DATA.exists() or DATA.read_text(encoding="utf-8") != text
        stale |= source.exists() and (not target.exists() or target.read_bytes() != source.read_bytes())
        print("site/ is stale: run scripts/build_site.py" if stale else "site/ is up to date")
        return int(stale)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(text, encoding="utf-8", newline="\n")
    if source.exists():
        shutil.copyfile(source, target)
    print(f"wrote {DATA.relative_to(REPO)} ({len(text) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
