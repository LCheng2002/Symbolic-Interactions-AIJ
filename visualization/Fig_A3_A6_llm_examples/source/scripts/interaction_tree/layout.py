"""Balance measured nodes and assign all coordinates before drawing."""

from .schema import PlacedNode

# Canvas width is the full AIJ text block (470.09 pt). Panel height is
# content-cropped per figure: header + tallest column + bottom margin, so
# the four stacked panels carry no blank bands between them.
PAGE_WIDTH = 470.09
SAFE_LEFT = 4.0
SAFE_RIGHT = PAGE_WIDTH - 4.0
HEADER_HEIGHT = 37.0
TREE_BOTTOM = 3.0
NODE_GAP = 3.0
_COL_LEFT = 6.0
_COL_RIGHT = PAGE_WIDTH - 6.0
_COL_GAP = 5.0
_N_COLUMNS = 5
_COL_W = (_COL_RIGHT - _COL_LEFT - (_N_COLUMNS - 1) * _COL_GAP) / _N_COLUMNS
COLUMN_BOUNDS = [
    (_COL_LEFT + i * (_COL_W + _COL_GAP), _COL_LEFT + i * (_COL_W + _COL_GAP) + _COL_W)
    for i in range(_N_COLUMNS)
]
N_COLUMNS = len(COLUMN_BOUNDS)
# Fig.1-style per-column trunks on each column's right edge.
TRUNK_X = [bounds[1] for bounds in COLUMN_BOUNDS]
COLUMN_CONTENT_WIDTH = _COL_W - 9.0


def assign_columns(measured_nodes):
    columns = [[] for _ in range(N_COLUMNS)]
    heights = [0.0] * N_COLUMNS
    for node in measured_nodes:
        column = min(range(N_COLUMNS), key=lambda index: (heights[index], index))
        columns[column].append(node)
        heights[column] += node.height + (NODE_GAP if len(columns[column]) > 1 else 0.0)
    return columns, heights


def place_nodes(measured_nodes, tree_top, tree_bottom=TREE_BOTTOM):
    columns, heights = assign_columns(measured_nodes)
    if max(heights, default=0.0) > tree_top - tree_bottom:
        raise ValueError(
            f"Tree is too tall at the fixed readable font size: {max(heights):.2f} pt"
        )
    placed = []
    for column_index, nodes in enumerate(columns):
        left, right = COLUMN_BOUNDS[column_index]
        cursor = tree_top
        for node in nodes:
            bottom = cursor - node.height
            placed.append(
                PlacedNode(
                    measured=node,
                    column=column_index,
                    left=left + 2.0,
                    right=right - 7.0,
                    top=cursor,
                    bottom=bottom,
                )
            )
            cursor = bottom - NODE_GAP
    placed.sort(key=lambda item: (item.column, -item.top))
    return placed, columns, heights
