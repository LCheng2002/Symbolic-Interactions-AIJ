from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Interaction:
    kind: str
    index: int
    value: float
    members: List[int]

    @property
    def key(self):
        return f"{self.kind}:{self.index}"


@dataclass(frozen=True)
class PlayerBox:
    member: int
    label: str
    lines: List[str]
    width: float
    height: float


@dataclass(frozen=True)
class MeasuredNode:
    interaction: Interaction
    label: str
    label_width: float
    label_height: float
    rows: List[List[PlayerBox]]
    row_heights: List[float]
    height: float


@dataclass(frozen=True)
class PlacedNode:
    measured: MeasuredNode
    column: int
    left: float
    right: float
    top: float
    bottom: float

