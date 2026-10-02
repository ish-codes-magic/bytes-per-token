"""Comparing two profiled servers call by call."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import compare_profiles  # noqa: E402


def profile(label, rows):
    host = [
        {"cat": cat, "name": name, "calls": calls, "self_ms": ms, "total_ms": ms}
        for cat, name, calls, ms in rows
    ]
    return {"label": label, "iterations": 10, "host": host}


SLOW = profile(
    "wg",
    [
        ("python_function", "split", 560, 40.0),
        ("python_function", "<built-in method acquire of _thread.lock object at 0x2aa3b>", 60, 300.0),
        ("python_function", "only here", 10, 5.0),
        ("Trace", "PyTorch Profiler (0)", 1, 999.0),
    ],
)
FAST = profile(
    "g",
    [
        ("python_function", "split", 560, 5.0),
        ("python_function", "<built-in method acquire of _thread.lock object at 0x2a6eb>", 60, 250.0),
    ],
)


def test_differences_are_per_step_and_ignore_addresses_and_the_profilers_own_rows():
    rows = compare_profiles.differences(SLOW, FAST)
    by_name = {row["name"]: row for row in rows}
    assert "PyTorch Profiler (0)" not in by_name
    lock = by_name["<built-in method acquire of _thread.lock>"]  # one row, although the addresses differ
    assert lock["diff_ms"] == 5.0 and lock["first"]["calls"] == 6.0
    assert by_name["split"]["diff_ms"] == 3.5  # (40 − 5) ms over 10 steps
    assert by_name["only here"]["second"] is None and by_name["only here"]["diff_ms"] == 0.5
    assert [row["name"] for row in rows][0] == "<built-in method acquire of _thread.lock>"  # largest first


def test_table():
    table = compare_profiles.table(SLOW, FAST, top=2)
    assert "| +3.50 | 4.00 ms × 56.0 | 0.50 ms × 56.0 | python_function | `split` |" in table
    assert "only here" not in table  # top 2 only
    assert "`wg`: self time × calls per step" in table
