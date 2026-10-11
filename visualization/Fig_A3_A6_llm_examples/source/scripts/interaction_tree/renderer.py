"""Draw a validated interaction tree on a fixed 235.045 x 318.045 point canvas.

Node style follows the manuscript's Fig. 1: light-blue filled player boxes
without outlines, a short dark-blue bar under each effect label, and a thin
vertical trunk on the outer edge of each column.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .layout import (
    COLUMN_BOUNDS,
    COLUMN_CONTENT_WIDTH,
    HEADER_HEIGHT,
    PAGE_WIDTH,
    SAFE_RIGHT,
    TREE_BOTTOM,
    TRUNK_X,
    assign_columns,
    place_nodes,
)
from .measure import (
    EFFECT_FONT_POINTS,
    PLAYER_FONT_POINTS,
    TextMeasurer,
    measure_node,
)
from .validator import validate

BLUE = "#28558A"
BOX_FILL = "#DBE5F1"
CENTER_X = PAGE_WIDTH / 2.0
STYLE = {
    "canvas_width_points": PAGE_WIDTH,
    "panel_height": "content-cropped: header + tallest column + bottom margin",
    "font_family": "DejaVu Sans",
    "effect_font_points": EFFECT_FONT_POINTS,
    "player_font_points": PLAYER_FONT_POINTS,
    "box_fill": BOX_FILL,
    "trunk": BLUE,
    "connector": BLUE,
    "columns": len(COLUMN_BOUNDS),
}


def _draw_node(ax, node):
    label_y = node.top
    ax.text(
        node.left,
        label_y,
        node.measured.label,
        fontsize=EFFECT_FONT_POINTS,
        fontfamily="DejaVu Serif",
        ha="left",
        va="top",
        color="black",
        zorder=4,
    )
    bar_y = label_y - node.measured.label_height - 1.2
    trunk_x = TRUNK_X[node.column]
    ax.plot(
        [min(node.left, trunk_x), max(node.left + node.measured.label_width, trunk_x)],
        [bar_y, bar_y],
        color=BLUE,
        lw=0.9,
        zorder=1,
    )
    cursor_y = bar_y - 3.0
    box_records = []
    for row_index, (row, row_height) in enumerate(zip(node.measured.rows, node.measured.row_heights)):
        total_width = sum(box.width for box in row) + 2.5 * max(0, len(row) - 1)
        x = node.left + max(0.0, (node.right - node.left - total_width) / 2.0)
        for box in row:
            y = cursor_y - box.height
            ax.add_patch(
                Rectangle(
                    (x, y),
                    box.width,
                    box.height,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=BOX_FILL,
                    zorder=3,
                )
            )
            ax.text(
                x + box.width / 2,
                y + box.height / 2,
                "\n".join(box.lines),
                fontsize=PLAYER_FONT_POINTS,
                fontweight="bold",
                fontfamily="DejaVu Sans",
                ha="center",
                va="center",
                linespacing=0.9,
                zorder=4,
            )
            if row_index == 0:
                ax.plot(
                    [x + box.width / 2, x + box.width / 2],
                    [bar_y, y + box.height],
                    color=BLUE,
                    lw=0.7,
                    zorder=1,
                )
            box_records.append(
                {
                    "member": box.member,
                    "label": box.label,
                    "lines": box.lines,
                    "bbox": [x, y, x + box.width, y + box.height],
                }
            )
            x += box.width + 2.5
        cursor_y -= row_height + 2.2
    return box_records


def render_tree(selected, players, prompt_label, prompt_lines, target, model_title, output_pdf, output_png):
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0,
        }
    )
    measurer = TextMeasurer(width_points=PAGE_WIDTH, height_points=PAGE_WIDTH)
    try:
        measured = [measure_node(item, players, measurer, COLUMN_CONTENT_WIDTH) for item in selected]
    finally:
        measurer.close()
    _, heights = assign_columns(measured)
    panel_height = HEADER_HEIGHT + max(heights, default=0.0) + TREE_BOTTOM
    tree_top = panel_height - HEADER_HEIGHT
    placed, columns, heights = place_nodes(measured, tree_top, TREE_BOTTOM)
    failures = validate(placed, selected, tree_top, TREE_BOTTOM)
    if failures:
        raise ValueError("Geometry validation failed: " + "; ".join(failures))

    fig = plt.figure(figsize=(PAGE_WIDTH / 72.0, panel_height / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_WIDTH)
    ax.set_ylim(0, panel_height)
    ax.axis("off")

    ax.text(10, panel_height - 6.0, prompt_label + ":", fontsize=8.0, fontweight="bold", va="top")
    ax.text(58, panel_height - 6.0, prompt_lines[0], fontsize=6.5, va="top")
    for offset, line in enumerate(prompt_lines[1:], start=1):
        ax.text(10, panel_height - 6.0 - 8.0 * offset, line, fontsize=6.5, va="top")
    ax.text(10, panel_height - 15.0, "Target:", fontsize=8.0, fontweight="bold", va="top")
    ax.text(58, panel_height - 15.0, target, fontsize=6.5, va="top")
    ax.text(SAFE_RIGHT, panel_height - 15.0, model_title, fontsize=8.5, fontweight="bold", ha="right", va="top")
    ax.text(
        CENTER_X,
        panel_height - 20.5,
        "+",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="#B8B8B8", lw=1.0),
        zorder=5,
    )

    node_meta = []
    for column_index, column_nodes in enumerate(columns):
        if not column_nodes:
            continue
        trunk_x = TRUNK_X[column_index]
        column_placed = [node for node in placed if node.column == column_index]
        trunk_bottom = min(
            node.top - node.measured.label_height - 1.2 for node in column_placed
        )
        ax.plot([CENTER_X, trunk_x], [panel_height - 25.5, panel_height - 33.0], color=BLUE, lw=1.2, zorder=0)
        ax.plot([trunk_x, trunk_x], [panel_height - 33.0, trunk_bottom], color=BLUE, lw=1.2, zorder=0)

    for node in placed:
        box_records = _draw_node(ax, node)
        node_meta.append(
            {
                "key": node.measured.interaction.key,
                "kind": node.measured.interaction.kind,
                "index": node.measured.interaction.index,
                "value": node.measured.interaction.value,
                "members": node.measured.interaction.members,
                "column": node.column,
                "bbox": [node.left, node.bottom, node.right, node.top],
                "boxes": box_records,
            }
        )

    fig.savefig(
        output_pdf,
        format="pdf",
        dpi=300,
        facecolor="white",
        metadata={
            "Creator": "A3-A6 unified interaction-tree renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_png, format="png", dpi=220, facecolor="white")
    plt.close(fig)
    return {
        "geometry_status": "PASS",
        "geometry_failures": failures,
        "style": STYLE,
        "panel_points": [PAGE_WIDTH, panel_height],
        "column_measured_heights": heights,
        "nodes": node_meta,
    }
