import pytest

pytest.importorskip("matplotlib")

from fastserve.viz.m6_figures import highlight_html, make_all  # noqa: E402

FIGURES = (
    "m6_highlight",
    "m6_accepted_lengths",
    "m6_theory",
    "m6_speedup_vs_users",
    "m6_speedup_surface",
    "m6_waterfall",
)


def test_every_figure_renders_with_a_data_driven_caption(m6_records, tmp_path):
    written = make_all(m6_records, tmp_path)
    names = {p.name for p in written}
    for figure in FIGURES:
        assert f"{figure}.png" in names, figure
    assert "m6_highlight.html" in names
    captions = {p.name: p.read_text(encoding="utf-8") for p in written if p.suffix == ".txt"}
    assert "Qwen3-0.6B drafted 62% of the" in captions["m6_highlight.caption.txt"]  # 5 of 8 tokens
    assert "accepts none on chat 40% of the time" in captions["m6_accepted_lengths.caption.txt"]
    assert (
        "EAGLE-3 head, k = 3, 1.80×) gives 0.90× at 64 users" in captions["m6_speedup_vs_users.caption.txt"]
    )
    assert "best cell is k = 3 at 1 user(s) (1.20×)" in captions["m6_speedup_surface.caption.txt"]
    assert "the worst k = 5 at 64 users (0.45×)" in captions["m6_speedup_surface.caption.txt"]
    assert "best method changes cost by -44%" in captions["m6_waterfall.caption.txt"]  # 1 / 1.8 − 1
    assert "+0%" in captions["m6_theory.caption.txt"]  # the fixture follows the formula exactly


def test_highlight_page_marks_each_token(m6_records):
    page = highlight_html(m6_records)
    assert '<span class="d">def</span>' in page and '<span class="t">(</span>' in page
    assert "Code, Qwen3-0.6B, k = 4: 62% drafted" in page
