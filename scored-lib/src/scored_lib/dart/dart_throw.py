from dataclasses import dataclass, field
import math
from scored_lib.dart.multiplier import Multiplier
from scored_lib.dart.constants import PolarPosition, Position


@dataclass(frozen=True, eq=True)
class DartThrow:
    """
    Represents a dart throw on the dartboard.

    Arguments:
        number (int): The number of the segment hit by the dart.
        multiplier (Multiplier): The normalized xy-coordinate relative to the center.
        position (Position): The position of the dart on the dartboard in normalized coordinates.
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
            return "Triple " + str(self.number)

    @property
    def polar(self) -> PolarPosition:
        """
        Returns the polar coordinates (r, theta) of the dart throw based on its position on the dartboard.

        r: normalized distance from center (0.0-1.0)

        θ: angle in radians, measured from positive x-axis, CCW

        :return: A tuple containing the polar coordinates (r, theta).
        """
        x, y = self.position
        dx, dy = x - 0.5, y - 0.5
        r = math.sqrt(dx * dx + dy * dy) / 0.5
        theta = math.atan2(dy, dx)
        if theta < 0:
            theta += 2 * math.pi
        return r, theta
