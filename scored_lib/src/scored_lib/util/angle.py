import math
from scored_lib.dart.constants import Position


def get_clockwise_angle(rel_pos: Position) -> float:
    """
    Convert normalized coordinates (0-1) to a clockwise angle (0-360).

    The angle starts at 0 degrees from the center-right (.5, 0)
    and increases clockwise. The center is assumed to be (0, 0).

    Parameters
    ----------
    x : Position
        The normalized xy-coordinate relative to the center.

    Returns
    -------
    float
        The angle in degrees from 0.0 to 360.0.
    """

    x, y = rel_pos

    angle = math.degrees(math.atan2(y, x))
    return (angle + 360) % 360
