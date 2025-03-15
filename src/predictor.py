from dataclasses import dataclass
from typing import List, Tuple
from board.dartboard import DartBoard, DartThrow, Position
import numpy as np
from perspective import compute_perspective, warp_point
from ultralytics import YOLO
import json
from util import BBYolo, compute_bounding_box


@dataclass(frozen=True)
class KeypointObject:
    name: str
    keypoints: List[Tuple[float, float]]
    bb: BBYolo
    conf: float

    def keypoints_x(self) -> List[float]:
        return [keypoint[0] for keypoint in self.keypoints]

    def keypoints_y(self) -> List[float]:
        return [keypoint[1] for keypoint in self.keypoints]


class DartPredictor:

    def __init__(self, board: DartBoard, model_path: str, conf: float):
        self._board = board
        self._dest_points = DartPredictor._get_perspective_transform_points(board)
        self.model = YOLO(model_path)
        self.conf = conf

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
        :return: The scores of each dart that was detected, the perspective transformation matrix and the detected objects.
        """

        res = self.model.predict(image)[0]

        res = json.loads(res.to_json())
        dartboard = None

        objects: List[KeypointObject] = [
            KeypointObject(obj["name"], obj["keypoints"], obj["confidence"])
            for obj in res
            if obj["confidence"] > self.conf
        ]
        for a in res:
            if a["confidence"] < self.conf:
                continue

            keypoints = zip(a["keypoints"]["x"], a["keypoints"]["y"])
            bb = compute_bounding_box(keypoints)
            obj = KeypointObject(a["name"], keypoints, bb, a["confidence"])
            objects.append(obj)

            if obj.name == "dartboard":
                dartboard = obj

        if dartboard is None:
            raise ValueError("No dartboard detected")

        src_points = dartboard.keypoints
        matrix = compute_perspective(src_points, self._dest_points)

        scores: List[DartThrow] = []

        for dart in [obj for obj in objects if obj.name == "dart"]:
            dart_tip = dart.keypoints[0]

            warped_point = warp_point(matrix, dart_tip)

            score = self._board.score_dart(warped_point)
            scores.append(score)

        return scores, matrix, objects
