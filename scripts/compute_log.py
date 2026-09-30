"""Write results/compute_log.csv from Modal's billing API: what this project's cloud compute actually cost.

    uv run --only-group local python scripts/compute_log.py

One row per (day, resource), in US dollars, for the app `bytes-per-token`. Modal reports complete hours only,
so the current day's rows grow as the day goes on: re-run to refresh.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START = "2026-09-29"  # the project's first cloud run
APP = "bytes-per-token"


def main() -> int:
    modal = Path(sys.executable).with_name("modal.exe" if os.name == "nt" else "modal")
    report = subprocess.run(
        [str(modal), "billing", "report", "--start", START, "-r", "h", "--show-resources", "--json"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    costs: dict[tuple[str, str], float] = defaultdict(float)
    last = ""
    for row in json.loads(report.stdout):
        if row["description"] != APP:
            continue
        costs[(row["interval_start"][:10], row["resource"])] += float(row["cost"])
        last = max(last, row["interval_start"])
    out = REPO / "results" / "compute_log.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["date", "resource", "usd", "provider", "billed_through_hour"])
        for (day, resource), usd in sorted(costs.items()):
            writer.writerow([day, resource, f"{usd:.4f}", "Modal", last])
    print(f"wrote {out.relative_to(REPO)}: ${sum(costs.values()):.2f} through {last} UTC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
