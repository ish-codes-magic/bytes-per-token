"""Fill generated blocks in Markdown files, so docs never contain hand-typed numbers.

A generated block looks like this, and everything between the two markers is replaced:

    <!-- BEGIN GENERATED: hw_summary -->
    ...
    <!-- END GENERATED: hw_summary -->
"""

from __future__ import annotations

import re
from pathlib import Path

_BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\r?\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def render(text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Replace the body of every block we have content for.

    Returns the new text and the names of blocks found in the text that had no content (left untouched).
    """
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in blocks:
            missing.append(name)
            return match.group(0)
        return f"{match.group(1)}{blocks[name].rstrip()}\n{match.group(4)}"

    return _BLOCK.sub(replace, text), missing


def render_file(path: str | Path, blocks: dict[str, str]) -> tuple[bool, list[str]]:
    """Render one file in place. Returns (changed, missing block names)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    new, missing = render(text, blocks)
    if new != text:
        path.write_text(new, encoding="utf-8", newline="\n")
    return new != text, missing
