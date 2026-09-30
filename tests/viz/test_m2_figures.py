import pytest

pytest.importorskip("matplotlib")

from fastserve.viz.m2_figures import FIGURES, make_all  # noqa: E402


def test_m2_figures_are_written_with_one_line_captions(m2_records, tmp_path):
    written = make_all(m2_records, tmp_path)
    for name in FIGURES:
        for suffix in (".png", ".svg", ".caption.txt"):
            assert tmp_path / f"{name}{suffix}" in written
        assert (tmp_path / f"{name}.caption.txt").read_text(encoding="utf-8").count("\n") == 1
    assert "up to 8 req/s" in (tmp_path / "m2_goodput.caption.txt").read_text(encoding="utf-8")
