"""Geometry and content validation for a fully placed tree."""

from .layout import SAFE_LEFT, SAFE_RIGHT


def overlap(a, b, tolerance=0.01):
    return not (
        a.right <= b.left + tolerance
        or b.right <= a.left + tolerance
        or a.bottom >= b.top - tolerance
        or b.bottom >= a.top - tolerance
    )


def validate(placed, selected, tree_top, tree_bottom):
    failures = []
    keys = [node.measured.interaction.key for node in placed]
    expected = [item.key for item in selected]
    if sorted(keys) != sorted(expected):
        failures.append("selected interactions were not rendered exactly once")
    if len(keys) != len(set(keys)):
        failures.append("duplicate rendered interaction")

    for node in placed:
        if node.left < SAFE_LEFT or node.right > SAFE_RIGHT:
            failures.append(f"{node.measured.interaction.key}: horizontal clipping")
        if node.top > tree_top or node.bottom < tree_bottom:
            failures.append(f"{node.measured.interaction.key}: vertical clipping")
        rendered_members = [box.member for row in node.measured.rows for box in row]
        if rendered_members != node.measured.interaction.members:
            failures.append(f"{node.measured.interaction.key}: member omission or reordering")

    for index, first in enumerate(placed):
        for second in placed[index + 1 :]:
            if overlap(first, second):
                failures.append(
                    f"node overlap: {first.measured.interaction.key}/{second.measured.interaction.key}"
                )
    return failures
