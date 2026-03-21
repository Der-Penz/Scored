import math

from scored_lib.dart.constants import Position


def get_clockwise_angle(pos: Position) -> float:
    """
    Convert normalized coordinates (0-1) to a clockwise angle (0-360).

    The angle starts at 0 degrees from the center-right (1.0, 0.5)
    and increases clockwise. The center is assumed to be (0.5, 0.5).

    Parameters
    ----------
    x : Position
        The normalized xy-coordinate (0.0 to 1.0).

    Returns
    -------
    float
        The angle in degrees from 0.0 to 360.0.
    """
    dx = pos[0] - 0.5
    dy = pos[1] - 0.5

    radians = math.atan2(dy, dx)
    degrees = math.degrees(radians)

    return degrees % 360
