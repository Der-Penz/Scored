import numpy as np

from scored_lib.dart.constants import DARTBOARD_NUMBERS, RING_DIMENSIONS, Position
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier


class DartBoard:
    """
    Dartboard using normalized coordinates in [0,1].
    (0.5, 0.5) is the center, outer circle has radius 0.5.
    """

    def __init__(self, ring_dimensions: dict[str, float] | None = None):
        if ring_dimensions is not None:
            self._rings = ring_dimensions.copy()
        else:
            self._rings = RING_DIMENSIONS.copy()

    @property
    def center(self) -> Position:
        return (0.5, 0.5)

    def get_rings(self) -> dict[str, float]:
        """Return normalized radius for all scoring regions."""
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
