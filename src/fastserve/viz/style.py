"""One visual style for every figure in the project (AGENTS.md §8).

- Colorblind-safe palette (Okabe–Ito), and never color alone: every series also gets its own marker.
- Each technique or number format keeps one color everywhere. The BF16 baseline is always gray.
- Every figure is saved as PNG (README) + SVG (docs), with a one-line takeaway caption next to it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are rendered in headless cloud containers
import matplotlib.pyplot as plt  # noqa: E402

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}
BASELINE_GRAY = "#6E6E6E"

# (color, marker) per series. Techniques and the number formats they use share a color.
SERIES = {
    "bf16": (BASELINE_GRAY, "o"),
    "baseline": (BASELINE_GRAY, "o"),
    "fp16": ("#A0A0A0", "h"),
    "fp8": (OKABE_ITO["blue"], "s"),
    "int8": (OKABE_ITO["sky_blue"], "D"),
    "int4": (OKABE_ITO["orange"], "^"),
    "kv_quant": (OKABE_ITO["green"], "v"),
    "prefix_cache": (OKABE_ITO["purple"], "P"),
    "speculative": (OKABE_ITO["vermillion"], "X"),
    "kernels": (OKABE_ITO["black"], "*"),
    # Hardware figures: measured vs promised.
    "copy": (OKABE_ITO["orange"], "o"),
    "read": (OKABE_ITO["blue"], "s"),
    "spec_sheet": (BASELINE_GRAY, None),
}


def series_style(name: str) -> dict[str, str]:
    color, marker = SERIES[name]
    return {"color": color, **({"marker": marker} if marker else {})}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 4.5),
            "figure.dpi": 100,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "savefig.bbox": "tight",
            "svg.hashsalt": "bytes-per-token",  # deterministic SVG ids -> clean git diffs
        }
    )


def save_figure(fig: plt.Figure, name: str, caption: str, out_dir: str | Path) -> list[Path]:
    """Write <name>.png, <name>.svg and <name>.caption.txt. Returns the paths written."""
    if "\n" in caption.strip():
        raise ValueError("a caption is one line: the figure's takeaway")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg", out / f"{name}.caption.txt"]
    fig.savefig(paths[0], dpi=200, metadata={"Software": None})
    fig.savefig(paths[1], metadata={"Date": None, "Creator": None})
    paths[2].write_text(caption.strip() + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    return paths
