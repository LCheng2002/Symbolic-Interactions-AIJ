"""Measure text and construct content-dependent interaction nodes."""

import re

import matplotlib.pyplot as plt

from .schema import Interaction, MeasuredNode, PlayerBox

EFFECT_FONT_POINTS = 5.3
PLAYER_FONT_POINTS = 5.3


class TextMeasurer:
    def __init__(self, width_points=720, height_points=500, dpi=100):
        self.dpi = dpi
        self.fig = plt.figure(figsize=(width_points / 72, height_points / 72), dpi=dpi)
        self.fig.canvas.draw()

    def close(self):
        plt.close(self.fig)

    def measure(self, text, fontsize, weight="normal", family="DejaVu Sans", linespacing=1.0):
        artist = self.fig.text(
            0,
            0,
            text,
            fontsize=fontsize,
            fontweight=weight,
            fontfamily=family,
            linespacing=linespacing,
            alpha=0.0,
        )
        self.fig.canvas.draw()
        bbox = artist.get_window_extent(renderer=self.fig.canvas.get_renderer())
        artist.remove()
        # Some interactive backends double the requested DPI on HiDPI displays.
        # Convert with the canvas' effective DPI so measurements stay in points.
        scale = 72.0 / self.fig.dpi
        return bbox.width * scale, bbox.height * scale


def wrap_label(label, measurer, max_text_width, fontsize=PLAYER_FONT_POINTS):
    words = label.split()
    if not words:
        return [label]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if measurer.measure(candidate, fontsize, weight="bold")[0] <= max_text_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if any(measurer.measure(line, fontsize, weight="bold")[0] > max_text_width for line in lines):
        raise ValueError(f"Player label cannot fit without arbitrary font shrinking: {label!r}")
    return lines


def interaction_label(interaction):
    return rf"$I_S^{{\mathrm{{{interaction.kind.lower()}}}}}$={interaction.value:.2f}"


def measure_node(interaction, players, measurer, content_width):
    label = interaction_label(interaction)
    label_width, label_height = measurer.measure(
        label, EFFECT_FONT_POINTS, family="DejaVu Serif"
    )
    boxes = []
    max_text_width = min(68.0, content_width - 10.0)
    for member in interaction.members:
        text = players[member]
        lines = wrap_label(text, measurer, max_text_width)
        rendered = "\n".join(lines)
        text_width, text_height = measurer.measure(
            rendered, PLAYER_FONT_POINTS, weight="bold", linespacing=0.9
        )
        boxes.append(
            PlayerBox(
                member=member,
                label=text,
                lines=lines,
                width=text_width + 5.0,
                height=text_height + 2.5,
            )
        )

    rows = []
    current = []
    used = 0.0
    for box in boxes:
        gap = 2.5 if current else 0.0
        if current and used + gap + box.width > content_width:
            rows.append(current)
            current = []
            used = 0.0
            gap = 0.0
        current.append(box)
        used += gap + box.width
    if current:
        rows.append(current)

    row_heights = [max(box.height for box in row) for row in rows]
    boxes_height = sum(row_heights) + 2.2 * max(0, len(rows) - 1)
    height = label_height + 1.2 + (3.0 if rows else 0.0) + boxes_height + 3.0
    return MeasuredNode(
        interaction=interaction,
        label=label,
        label_width=label_width,
        label_height=label_height,
        rows=rows,
        row_heights=row_heights,
        height=height,
    )
