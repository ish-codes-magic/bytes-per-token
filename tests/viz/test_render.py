"""The shared figure renderer: which raw files it reads, and how a stale figure is detected."""

import json

import pytest

from fastserve.viz.render import available, load_raw, render, stale_captions


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def record(run_id, experiment="probe"):
    return {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp": f"2026-10-0{run_id}T00:00:00",
        "metrics": {},
    }


def test_load_raw_keeps_the_newest_run_of_a_campaign_and_every_run_of_rerunnable_tasks(tmp_path):
    write(tmp_path / "hw_probe.jsonl", [record(1), record(1), record(2)])
    write(tmp_path / "m8_ablation.jsonl", [record(1, "serving"), record(2, "serving")])
    raw = load_raw(tmp_path)
    assert set(raw) == {"hw", "m8"}
    assert [r["run_id"] for r in raw["hw"]] == [2]  # one campaign: its newest run
    assert [r["run_id"] for r in raw["m8"]] == [1, 2]  # tasks re-run one at a time: all of them
    assert available(raw) == ["m0", "m8"]  # m0 is drawn from the hardware probe
    assert load_raw(tmp_path / "nothing-here") == {}


def test_a_caption_that_changed_marks_its_figure_stale(tmp_path):
    fresh, committed = tmp_path / "fresh", tmp_path / "committed"
    fresh.mkdir()
    committed.mkdir()
    for name, old, new in (("same", "a\n", "a"), ("changed", "1.5x", "1.6x")):
        (committed / f"{name}.caption.txt").write_text(old, encoding="utf-8")
        (fresh / f"{name}.caption.txt").write_text(new, encoding="utf-8")
    (fresh / "new.caption.txt").write_text("never committed", encoding="utf-8")
    (fresh / "same.png").write_bytes(b"")
    assert stale_captions(sorted(fresh.iterdir()), committed) == ["changed", "new"]


def test_unknown_milestone():
    with pytest.raises(ValueError, match="unknown milestone"):
        render("m99", {}, ".", ".")
