type Position = tuple[int, int]
type PolarPosition = tuple[float, float]

RING_SIZE_MM = 8
BOARD_RADIUS_MM = 451 / 2

BED_ANGLE_DEGREES = 360 / 20

# Normalized ring radius values based on the official dartboard dimensions
RING_RADIUS_NORMALIZED = {
    "inner_bull": (12.7 / 2) / BOARD_RADIUS_MM,
    "outer_bull": (32 / 2) / BOARD_RADIUS_MM,
    "triple_inner": (214 / 2 - 8) / BOARD_RADIUS_MM,
    "triple_outer": (214 / 2) / BOARD_RADIUS_MM,
    "double_inner": (340 / 2 - 8) / BOARD_RADIUS_MM,
    "double_outer": (340 / 2) / BOARD_RADIUS_MM,
    "edge": BOARD_RADIUS_MM / BOARD_RADIUS_MM,
}

DARTBOARD_NUMBERS: list[int] = [
    1,
    18,
    4,
    13,
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
]
