from __future__ import annotations

from typing import Tuple

import numpy as np
from lib.scored.darts.DartThrow import DartThrow
from lib.scored.darts.Multiplier import Multiplier

RING_SIZE = 8
BOARD_RADIUS_NORMALIZED = 451 / 2

# Normalized dimensions
RING_DIMENSIONS = {
    "inner_bull": (12.7 / 2) / BOARD_RADIUS_NORMALIZED,
    "outer_bull": (32 / 2) / BOARD_RADIUS_NORMALIZED,
    "triple_inner": (214 / 2 - 8) / BOARD_RADIUS_NORMALIZED,
    "triple_outer": (214 / 2) / BOARD_RADIUS_NORMALIZED,
    "double_inner": (340 / 2 - 8) / BOARD_RADIUS_NORMALIZED,
    "double_outer": (340 / 2) / BOARD_RADIUS_NORMALIZED,
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
    Dartboard using normalized coordinates in [0,1].
    (0.5, 0.5) is the center, outer circle has radius 0.5.
    """

    def __init__(self):
        self._rings = RING_DIMENSIONS.copy()

    def get_center(self) -> Position:
        return (0.5, 0.5)

    def get_rings(self) -> dict[str, float]:
        """Return normalized radii for all scoring regions."""
        return self._rings

    def score_dart(self, position: Position) -> DartThrow:
        """
        Retrieves the score of a dart based on the position on the dartboard.

        :param position: The position of the dart throw in normalized coordinates.
        :return: The resulting DartThrow object with score and position.
        """
        number = self._get_dart_number(position)
        multiplier = self._get_dart_multiplier(position)
        return DartThrow(number, multiplier, position)

    def _get_dart_number(self, position: Position):
        dx, dy = position[0] - 0.5, position[1] - 0.5
        angle = np.arctan2(dy, dx)

        if angle < 0:
            angle += 2 * np.pi

        segment_angle = (2 * np.pi) / 20
        segment_index = int(angle // segment_angle)

        return DARTBOARD_NUMBERS[segment_index]

    def _get_dart_multiplier(self, position: Position) -> Multiplier:
        dx, dy = position[0] - 0.5, position[1] - 0.5
        distance = np.sqrt(dx * dx + dy * dy) / 0.5

        if distance <= self._rings["inner_bull"]:
            return Multiplier.INNER_BULL
        elif distance <= self._rings["outer_bull"]:
            return Multiplier.OUTER_BULL
        elif self._rings["triple_inner"] <= distance <= self._rings["triple_outer"]:
            return Multiplier.TRIPLE
        elif self._rings["double_inner"] <= distance <= self._rings["double_outer"]:
            return Multiplier.DOUBLE
        elif distance <= 1.0:
            return Multiplier.SINGLE
        else:
            return Multiplier.MISS
