import math

from scored_lib.dart.constants import (
    DARTBOARD_NUMBERS,
    RING_RADIUS_NORMALIZED,
    BED_ANGLE_DEGREES,
    Position,
)
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier
from scored_lib.util.angle import get_clockwise_angle


def get_scored_number(rel_pos: Position) -> int:
    """
    Get the scored number based on the position on the dartboard.

    Parameters
    ----------
    rel_pos : Position
        The normalized xy-coordinate relative to the center.

    Returns
    -------
    int
        The scored number corresponding to the position. 0 for misses.
    """
    angle = get_clockwise_angle(rel_pos)

    # Shift angle by half a bed (8 degrees) to align with boundaries
    angle = (angle + BED_ANGLE_DEGREES / 2) % 360

    offset = 4  # Offset to align with the numbers
    index = int(angle // BED_ANGLE_DEGREES) + offset

    return DARTBOARD_NUMBERS[index % len(DARTBOARD_NUMBERS)]


def get_scored_multiplier(rel_pos: Position) -> Multiplier:
    """
    Get the scored multiplier based on the position on the dartboard.

    Parameters
    ----------
    rel_pos : Position
        The normalized xy-coordinate relative to the center.

    Returns
    -------
    Multiplier
        The scored multiplier corresponding to the position.
    """
    distance = math.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)

    if distance <= RING_RADIUS_NORMALIZED["inner_bull"]:
        return Multiplier.INNER_BULL
    elif distance <= RING_RADIUS_NORMALIZED["outer_bull"]:
        return Multiplier.OUTER_BULL
    elif (
        RING_RADIUS_NORMALIZED["triple_inner"] < distance <= RING_RADIUS_NORMALIZED["triple_outer"]
    ):
        return Multiplier.TRIPLE
    elif (
        RING_RADIUS_NORMALIZED["double_inner"] < distance <= RING_RADIUS_NORMALIZED["double_outer"]
    ):
        return Multiplier.DOUBLE
    else:
        return Multiplier.SINGLE


def score_dart_throw(rel_pos: Position) -> DartThrow:
    """
    Score a dart throw based on its position on the dartboard.

    Parameters
    ----------
    rel_pos : Position
        The normalized xy-coordinate relative to the center.

    Returns
    -------
    DartThrow
        An object containing the scored number and multiplier.
    """
    number = get_scored_number(rel_pos)
    multiplier = get_scored_multiplier(rel_pos)

    return DartThrow(number=number, multiplier=multiplier, position=rel_pos)
