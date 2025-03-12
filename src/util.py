from dataclasses import dataclass
import sys
from typing import Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class BBYolo:
    """
    A bounding box with a center, width and height. All values are in the range [0, 1].
    """

    center: Tuple[float, float]
    width: float
    height: float


def compute_bounding_box(
    key_points: Sequence[Tuple[float, float]], padding: float = 0
) -> BBYolo:
    """
    Computes a bounding box around the given keypoints with padding.

    :param key_points: The keypoints to compute the bounding box around
    :param padding: The padding in percentage of the image size
    :return: A bounding box in YOLO format
    """

    if len(key_points) == 0:
        raise ValueError("No keypoints provided")

    if len(key_points) == 1:
        print(
            "Only one keypoint provided, returning a bounding box with width and height 0."
        )
        return BBYolo(key_points[0], 0, 0)

    x, y = np.stack(key_points).T
    min_x, min_y = max(np.min(x) - padding, 0), max(np.min(y) - padding, 0)
    max_x, max_y = min(np.max(x) + padding, 1), min(np.max(y) + padding, 1)

    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    width, height = max_x - min_x, max_y - min_y
    return BBYolo(center, width, height)


def to_absolute_pixels(key_point: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Converts a keypoint from relative coordinates to absolute pixel coordinates.

    :param key_point: The keypoint in relative coordinates
    :param shape: The shape of the image
    :return: The keypoint in absolute pixel coordinates
    """
    return np.array([key_point[0] * shape[0], key_point[1] * shape[1]])


def loading_bar(iteration: int, total: int, length: int = 30):
    """
    Prints a loading bar to the console.

    :param iteration: The current iteration
    :param total: The total number of iterations
    :param length: The length of the loading bar
    """
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)

    sys.stdout.write(f"\r[{bar}] {percent:.1f}%")
    sys.stdout.flush()
