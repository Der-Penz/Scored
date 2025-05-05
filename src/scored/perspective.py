from typing import Tuple, Sequence
import numpy as np
import cv2


def compute_perspective(
    src: Sequence[Tuple[float, float]],
    dest: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """
    Compute the perspective transform matrix

    :param src: source coordinates
    :param dest: destination coordinates
    :param dsize: size of the destination image
    :return: perspective transform matrix
    """
    src = np.array(src, np.float32)
    dest = np.array(dest, np.float32)
    matrix = cv2.getPerspectiveTransform(src, dest)
    return matrix


def warp_point(M: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
    """
    Applies a perspective transformation to a single point using matrix M.

    :param M: 3x3 perspective transformation matrix
    :param point: (x, y) coordinates of the point in the original image
    :return: (x', y') coordinates of the point in the warped space
    """
    point_homogeneous = np.array([[point[0], point[1], 1]], dtype=np.float32)
    transformed_point = np.dot(M, point_homogeneous.T).T

    x_warped = transformed_point[0, 0] / transformed_point[0, 2]
    y_warped = transformed_point[0, 1] / transformed_point[0, 2]

    return x_warped, y_warped


def warp_image(M: np.ndarray, image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Applies a perspective transformation to an image using matrix M.

    :param M: 3x3 perspective transformation matrix
    :param image: input image
    :param shape: shape of the output image
    :return: warped image
    """

    return cv2.warpPerspective(image, M, shape)
