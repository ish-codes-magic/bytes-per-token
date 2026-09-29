import pytest

pytest.importorskip("matplotlib")

from fastserve.viz.hw_figures import FIGURES, make_all  # noqa: E402


def test_make_all_writes_png_svg_and_caption(probe_records, tmp_path):
    written = make_all(probe_records, tmp_path)
    for name in FIGURES:
        for suffix in (".png", ".svg", ".caption.txt"):
            path = tmp_path / f"{name}{suffix}"
            assert path in written and path.stat().st_size > 0


def test_captions_are_one_line_takeaways_built_from_data(probe_records, tmp_path):
    make_all(probe_records, tmp_path)
    captions = {name: (tmp_path / f"{name}.caption.txt").read_text(encoding="utf-8") for name in FIGURES}
    assert all(c.count("\n") == 1 for c in captions.values())
    assert "280 GB/s" in captions["hw_bandwidth_vs_size"]  # the fake 1 GiB read speed
    assert "93%" in captions["hw_bandwidth_vs_size"]  # 280 / 300
    assert "ridge point" in captions["hw_roofline"]


def test_svg_output_is_deterministic(probe_records, tmp_path):
    make_all(probe_records, tmp_path / "a")
    make_all(probe_records, tmp_path / "b")
    for name in FIGURES:
        assert (tmp_path / "a" / f"{name}.svg").read_bytes() == (tmp_path / "b" / f"{name}.svg").read_bytes()
