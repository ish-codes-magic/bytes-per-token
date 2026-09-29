from fastserve.report.render import render, render_file

DOC = """# Title

<!-- BEGIN GENERATED: table -->
old content
<!-- END GENERATED: table -->

<!-- BEGIN GENERATED: later -->
<!-- END GENERATED: later -->
"""


def test_render_replaces_known_blocks_and_reports_missing_ones():
    new, missing = render(DOC, {"table": "| a |\n|---|"})
    assert "old content" not in new
    assert "<!-- BEGIN GENERATED: table -->\n| a |\n|---|\n<!-- END GENERATED: table -->" in new
    assert missing == ["later"]


def test_render_is_idempotent():
    once, _ = render(DOC, {"table": "x", "later": "y"})
    twice, _ = render(once, {"table": "x", "later": "y"})
    assert once == twice


def test_render_file_only_writes_on_change(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(DOC, encoding="utf-8")
    assert render_file(path, {"table": "x"})[0] is True
    assert render_file(path, {"table": "x"})[0] is False
