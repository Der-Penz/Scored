from typing import List
from board.dartboard import DartBoard, DartThrow, Position
import numpy as np
from perspective import compute_perspective, warp_point
from ultralytics import YOLO
import json


class DartPredictor:

    def __init__(self, board: DartBoard, model_path: str):
        self._board = board
        self._dest_points = DartPredictor._get_perspective_transform_points(board)
        print(self._dest_points)
        self.model = YOLO(model_path)

    @staticmethod
    def _get_perspective_transform_points(board: DartBoard) -> List[Position]:
        half = board.get_size() // 2
        center = board.get_center()

        points = [
            (center[0], center[1] - half),
            (center[0] + half, center[1]),
            (center[0], center[1] + half),
            (center[0] - half, center[1]),
        ]
        return points

    def predict(self, image: np.ndarray) -> List[DartThrow]:
        """
        Predicts the position of a dart throw based on the given image.

        :param image: The image of a dartboard which will be used for prediction.
        :return: The scores of each dart that was detected and the perspective transformation matrix.
        """

        res = self.model.predict(image)[0]

        res = json.loads(res.to_json())
        dartboard = None
        darts = []
        for obj in res:
            if obj["name"] == "dartboard":
                dartboard = obj
            elif obj["name"] == "dart":
                darts.append(obj)

        x, y = dartboard["keypoints"]["x"], dartboard["keypoints"]["y"]
        src_points = [pos for pos in zip(x, y)]

        matrix = compute_perspective(src_points, self._dest_points)

        scores = []
        for dart in darts:
            dart_tip = dart["keypoints"]["x"][0], dart["keypoints"]["y"][0]
            warped_point = warp_point(matrix, dart_tip)

            score = self._board.score_dart(warped_point)
            scores.append(score)

        return scores, matrix
