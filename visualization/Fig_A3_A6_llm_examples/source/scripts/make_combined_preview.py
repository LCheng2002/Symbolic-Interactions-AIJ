#!/usr/bin/env python
"""Compose the four A3--A6 panels into one 4x1 page-fill preview PNG.

The combined canvas is exactly the manuscript text block
(470.09 x 636.09 pt) so this preview shows the four panels at their
final printed size.
"""

import subprocess
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
OUTPUTS = ROOT.parent / "outputs"
DPI = 300
TEXT_BLOCK_POINTS = (470.09, 636.09)
STEMS = [
    "AIJ-FigA3-deepseek",
    "AIJ-FigA4-deepseek",
    "AIJ-FigA5-qwen",
    "AIJ-FigA6-qwen",
]


def main():
    panels = []
    for stem in STEMS:
        prefix = OUTPUTS / f"_preview_{stem}"
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-r", str(DPI), "-png",
             str(OUTPUTS / f"{stem}.pdf"), str(prefix)],
            check=True,
        )
        panel = Image.open(f"{prefix}.png").convert("RGB")
        Path(f"{prefix}.png").unlink()
        panels.append(panel)
    width = round(TEXT_BLOCK_POINTS[0] / 72.0 * DPI)
    combined = Image.new("RGB", (width, sum(p.height for p in panels)), "white")
    cursor = 0
    for panel in panels:
        combined.paste(panel, (0, cursor))
        cursor += panel.height
    out = OUTPUTS / "AIJ-FigA3-A6-combined-preview.png"
    combined.save(out)
    print(f"{out} {combined.size}")


if __name__ == "__main__":
    main()
