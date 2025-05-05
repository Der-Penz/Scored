from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Tuple

RING_SIZE = 8

DIMENSIONS = {
    "inner_bull": 12.7 / 2,
    "outer_bull": 32 / 2,
    "double_inner": 340 / 2 - RING_SIZE,
    "double_outer": 340 / 2,
    "triple_inner": 214 / 2 - RING_SIZE,
    "triple_outer": 214 / 2,
    "radius": 451 / 2,
}

DARTBOARD_NUMBERS = [
    6,
    10,
    15,
    2,
    17,
    3,
    19,
    7,
    16,
    8,
    11,
    14,
    9,
    12,
    5,
    20,
    1,
    18,
    4,
    13,
]

type Position = Tuple[int, int]
type ScoredDart = Tuple[int, int]


class DartBoard:
    """
    Represents a dartboard with a given resolution.
    The resolution defines how accurate reading the dartboard is.
    """

    def __init__(self, resolution: int):
        self._size = resolution
        self._center: Position = (resolution // 2, resolution // 2)
        self._rings = DartBoard._get_rings(resolution)

    def get_size(self) -> int:
        return self._size

    def shape(self) -> Tuple[int, int]:
        return (self._size, self._size)

    def get_center(self) -> Position:
        return self._center

    def get_rings(self) -> dict[str, float]:
        return self._rings

    @staticmethod
    def _get_rings(size: int) -> dict[str, float]:
        scale_factor = (size / 2) / DIMENSIONS["double_outer"]

        rings = {
            "inner_bull": DIMENSIONS["inner_bull"] * scale_factor,
            "outer_bull": DIMENSIONS["outer_bull"] * scale_factor,
            "double_inner": DIMENSIONS["double_inner"] * scale_factor,
            "double_outer": DIMENSIONS["double_outer"] * scale_factor,
            "triple_outer": DIMENSIONS["triple_outer"] * scale_factor,
            "triple_inner": DIMENSIONS["triple_inner"] * scale_factor,
        }

        return rings

    def score_dart(self, position: Position, relative=False) -> DartThrow:
        """
        Retrieves the score of a dart throw based on the position on the dartboard.

        :param position: The position of the dart throw.
        :param relative: If True, the position is relative coordinates.
        :return: The score of the dart
        """
        if relative:
            position = (position[0] + self._size, position[1] + self._size)

        number = self._get_dart_number(position)
        multiplier = self._get_dart_multiplier(position)
        return DartThrow(number, multiplier, position)

    def _get_dart_number(self, position: Position):
        segment_angle = (2 * np.pi) / 20

        rel_position = np.array(position) - np.array(self._center)
        angle = np.arctan2(rel_position[1], rel_position[0])

        if angle < 0:
            angle += 2 * np.pi

        segment_index = int(angle // segment_angle)

        return DARTBOARD_NUMBERS[segment_index]

    def _get_dart_multiplier(self, position: Position) -> Multiplier:
        distance_from_center = np.linalg.norm(
            np.array(position) - np.array(self._center)
        )

        if distance_from_center <= self._rings["inner_bull"]:
            return Multiplier.INNER_BULL
        elif distance_from_center <= self._rings["outer_bull"]:
            return Multiplier.OUTER_BULL
        elif (
            distance_from_center <= self._rings["triple_outer"]
            and distance_from_center >= self._rings["triple_inner"]
        ):
            return Multiplier.TRIPLE
        elif (
            distance_from_center <= self._rings["double_outer"]
            and distance_from_center >= self._rings["double_inner"]
        ):
            return Multiplier.DOUBLE
        elif distance_from_center <= self._rings["double_outer"]:
            return Multiplier.SINGLE
        else:
            return Multiplier.MISS


class Multiplier(Enum):
    MISS = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    INNER_BULL = 4
    OUTER_BULL = 5


@dataclass(frozen=True, eq=True)
class DartThrow:
    """
    Represents a dart throw on the dartboard.

    Arguments:
        number (int): The number of the segment hit by the dart.
        multiplier (Multiplier): The multiplier for the score.
        position (Position): The position of the dart on the dartboard.
    """

    number: int
    multiplier: Multiplier
    position: Position = field(compare=False)

    @property
    def score(self) -> int:
        if self.multiplier == Multiplier.MISS:
            return 0
        if self.multiplier == Multiplier.INNER_BULL:
            return 50
        if self.multiplier == Multiplier.OUTER_BULL:
            return 25
        return self.number * self.multiplier.value

    @property
    def is_bull(self) -> bool:
        return self.multiplier in (Multiplier.INNER_BULL, Multiplier.OUTER_BULL)

    @property
    def is_double(self) -> bool:
        return self.multiplier == Multiplier.DOUBLE

    @property
    def is_triple(self) -> bool:
        return self.multiplier == Multiplier.TRIPLE

    @property
    def is_miss(self) -> bool:
        return self.multiplier == Multiplier.MISS

    @property
    def is_single(self) -> bool:
        return self.multiplier == Multiplier.SINGLE

    @property
    def short_label(self) -> str:
        if self.multiplier == Multiplier.MISS:
            return "X"
        if self.multiplier == Multiplier.INNER_BULL:
            return "50"
        if self.multiplier == Multiplier.OUTER_BULL:
            return "25"
        if self.multiplier == Multiplier.SINGLE:
            return str(self.number)
        if self.multiplier == Multiplier.DOUBLE:
            return "D" + str(self.number)
        if self.multiplier == Multiplier.TRIPLE:
            return "T" + str(self.number)

    @property
    def label(self) -> str:
        if self.multiplier == Multiplier.MISS:
            return "MISS"
        if self.multiplier == Multiplier.INNER_BULL:
            return "BULL"
        if self.multiplier == Multiplier.OUTER_BULL:
            return "OUTER BULL"
        if self.multiplier == Multiplier.SINGLE:
            return "Single " + str(self.number)
        if self.multiplier == Multiplier.DOUBLE:
            return "Double " + str(self.number)
        if self.multiplier == Multiplier.TRIPLE:
            return "Triple" + str(self.number)
