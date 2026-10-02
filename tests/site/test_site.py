"""The dashboard: its data file is current, and the browser's model reproduces the Python model."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from fastserve.report.site import _grid, rounded  # noqa: E402


def same(a, b, tolerance: float = 1e-6) -> bool:
    """Equal structures, floats compared to a relative tolerance (platforms round the last digit apart)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key], tolerance) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tolerance) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or isinstance(a, str) or isinstance(b, str):
            return False
        return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-300)
    return a == b


def test_rounding_makes_a_file_that_is_the_same_everywhere():
    assert rounded({"a": [1.23456789012, 2], "b": float("nan"), "c": "x"}) == {
        "a": [1.234568, 2],
        "b": None,
        "c": "x",
    }
    assert rounded(0.1 + 0.2, digits=12) == 0.3


def test_a_needle_grid_is_the_share_of_secrets_found_per_cell():
    cells = [
        {"length": 1024, "depth": 0.0, "passed": True},
        {"length": 1024, "depth": 0.0, "passed": False},
        {"length": 4096, "depth": 0.0, "passed": True},
        {"length": 1024, "depth": 0.5, "passed": False},
        {"length": 4096, "depth": 0.5, "passed": True},
    ]
    grid = _grid("demo", cells)
    assert grid["lengths"] == [1024, 4096] and grid["depths"] == [0.0, 0.5]
    assert grid["passed"] == [[0.5, 1.0], [0.0, 1.0]]  # rows are depths, columns are lengths


def test_the_committed_data_file_is_what_the_raw_results_give():
    import build_site

    committed = json.loads(build_site.DATA.read_text(encoding="utf-8"))
    fresh = json.loads(build_site.build())
    assert list(fresh) == list(committed)
    stale = [key for key in fresh if not same(fresh[key], committed[key])]
    assert not stale, f"site/data/dashboard.json is stale in {stale}: run scripts/build_site.py"
    source, target = build_site.HIGHLIGHT
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("script", ["parity.mjs", "logic.mjs"])
def test_the_browsers_model_and_logic(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed here; these run in CI and on the laptop")
    result = subprocess.run(
        [node, str(REPO / "site" / "tests" / script)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
