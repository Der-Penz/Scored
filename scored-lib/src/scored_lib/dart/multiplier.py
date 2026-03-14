from enum import Enum


class Multiplier(Enum):
    """
    Represents the segment a dart hits on the dartboard, which determines the score multiplier.
    """

    MISS = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    INNER_BULL = 4
    OUTER_BULL = 5
