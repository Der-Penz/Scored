from scored_lib.dart.constants import BED_ANGLE_DEGREES, DARTBOARD_NUMBERS, RING_RADIUS_NORMALIZED, Position
from scored_lib.dart.multiplier import Multiplier
import math

def get_segment_default_position(number: int, multiplier: Multiplier) -> Position:
    """
    Returns the default position of a dart for a given segment

    Parameters
    ----------
    number : int
        The number of the segment hit by the throw.
    multiplier : Multiplier
        The multiplier of the segment hit by the throw.
    Returns
    -------
    Position
        The (x, y) position in relative coordinates
    """
    if multiplier is Multiplier.INNER_BULL:
        radius, angle_rad = 0.0, 0.0
    elif multiplier is Multiplier.OUTER_BULL:
        radius = (RING_RADIUS_NORMALIZED["inner_bull"] + RING_RADIUS_NORMALIZED["outer_bull"]) / 2
        angle_rad = 0.0
    elif multiplier is Multiplier.MISS:
        radius = (RING_RADIUS_NORMALIZED["double_outer"] + RING_RADIUS_NORMALIZED["edge"]) / 2
        angle_rad = 0.0
    else:
        
        if multiplier == Multiplier.SINGLE:
            inner_key, outer_key = "triple_outer", "double_inner"
        elif multiplier == Multiplier.DOUBLE:
            inner_key, outer_key = "double_inner", "double_outer"
        elif multiplier == Multiplier.TRIPLE:
            inner_key, outer_key = "triple_inner", "triple_outer"
            
        radius = (RING_RADIUS_NORMALIZED[inner_key] + RING_RADIUS_NORMALIZED[outer_key]) / 2
        angle_rad = math.radians(((DARTBOARD_NUMBERS.index(number) - 4) * BED_ANGLE_DEGREES) % 360)

    return (
        radius * math.cos(angle_rad),
        radius * math.sin(angle_rad),
    )


def canvas_to_relative_position(position: Position) -> Position:
    """Convert canvas normalized (0-1, center 0.5) coords to the scoring frame."""
    return ((position[0] - 0.5) / 0.5, (position[1] - 0.5) / 0.5)

def relative_to_canvas_position(position: Position) -> Position:
    """Convert scoring frame coords to canvas normalized (0-1, center 0.5) coords."""
    return (position[0] * 0.5 + 0.5, position[1] * 0.5 + 0.5)