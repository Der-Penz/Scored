from __future__ import annotations
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
        return DartThrow(number, multiplier)

    def _get_dart_number(self, position: Position):
        segment_angle = (2 * np.pi) / 20

        rel_position = np.array(position) - np.array(self._center)
        angle = np.arctan2(rel_position[1], rel_position[0])

        if angle < 0:
            angle += 2 * np.pi
            
        segment_index = int(angle // segment_angle)

        return DARTBOARD_NUMBERS[segment_index]

    def _get_dart_multiplier(self, position: Position) -> int:
        distance_from_center = np.linalg.norm(
            np.array(position) - np.array(self._center)
        )

        if distance_from_center <= self._rings["inner_bull"]:
            return 1
        elif distance_from_center <= self._rings["outer_bull"]:
            return 2
        elif (
            distance_from_center <= self._rings["triple_outer"]
            and distance_from_center >= self._rings["triple_inner"]
        ):
            return 5
        elif (
            distance_from_center <= self._rings["double_outer"]
            and distance_from_center >= self._rings["double_inner"]
        ):
            return 4
        elif distance_from_center <= self._rings["double_outer"]:
            return 3
        else:
            return 0


class DartThrow:
    """
    Represents a dart throw with a given number and multiplier.
    """

    def __init__(self, number: int, multiplier: int):
        self._number = number
        self._multiplier = multiplier

    def score(self) -> int:
        if self._multiplier == 0:
            return 0
        if self._multiplier == 1:
            return 50
        if self._multiplier == 2:
            return 25

        x = [1, 2, 3][self._multiplier - 3]
        return self._number * x

    def short_label(self) -> str:
        if self._multiplier == 0:
            return "X"
        if self._multiplier == 1:
            return "50"
        if self._multiplier == 2:
            return "25"

        x = ["", "D", "T"][self._multiplier - 3]
        return x + str(self._number)

    def label(self) -> str:
        if self._multiplier == 0:
            return "MISS"
        if self._multiplier == 1:
            return "BULL"
        if self._multiplier == 2:
            return "OUTER BULL"

        x = ["Single", "Double", "Triple"][self._multiplier - 3]
        return f"{x} {str(self._number)}"
