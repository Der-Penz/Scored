from dataclasses import dataclass
from typing import List, Tuple
from board.dartboard import DartBoard, DartThrow, Position
import numpy as np
from perspective import compute_perspective, warp_point
from ultralytics import YOLO
import json


@dataclass(frozen=True)
class KeypointObject:
    name: str
    keypoints: List[Tuple[float, float]]
    bb: Tuple[Tuple[float, float], float, float]
    conf: float

    def keypoints_x(self) -> List[float]:
        return [keypoint[0] for keypoint in self.keypoints]

    def keypoints_y(self) -> List[float]:
        return [keypoint[1] for keypoint in self.keypoints]


@dataclass(frozen=True)
class DartPrediction:
    scores: List[DartThrow]
    matrix: np.ndarray
    objects: List[KeypointObject]

    def sum_score(self) -> int:
        """
        Returns the sum of all scores.
        """
        return sum([score.score() for score in self.scores])
    
    def get_dartboard(self) -> KeypointObject:
        """
        Returns the dartboard object.
        """
        for obj in self.objects:
            if obj.name == "dartboard":
                return obj
        raise ValueError("No dartboard detected")
    
    def get_darts(self) -> List[KeypointObject]:
        """
        Returns the dart objects.
        """
        return [obj for obj in self.objects if obj.name == "dart"]


class DartPredictor:

    def __init__(self, board: DartBoard, model_path: str, conf: float):
        self._board = board
        self._dest_points = DartPredictor.get_perspective_transform_points(board)
        self.model: YOLO = YOLO(model_path)
        self.conf = conf

    def model(self):
        return self.model

    @staticmethod
    def get_perspective_transform_points(board: DartBoard) -> List[Position]:
        half = board.get_size() // 2
        center = board.get_center()

        points = [
            (center[0], center[1] - half),
            (center[0] + half, center[1]),
            (center[0], center[1] + half),
            (center[0] - half, center[1]),
        ]
        return points

    def predict(
        self, image: np.ndarray, conf: float = None, **kwargs
    ) -> DartPrediction:
        """
        Predicts the position of a dart throw based on the given image.

        :param image: The image of a dartboard which will be used for prediction.
        :param conf: The confidence threshold for the model. If None, the default confidence threshold will be used.
        :param kwargs: Additional arguments to be passed to the model.
        :return: The scores of each dart that was detected, the perspective transformation matrix and the detected objects.
        """

        res = self.model.predict(image, **kwargs)[0]

        res = json.loads(res.to_json())
        dartboard = None

        objects: List[KeypointObject] = []
        for a in res:
            t_conf = conf if conf is not None else self.conf
            if a["confidence"] < t_conf:
                continue

            keypoints = list(zip(a["keypoints"]["x"], a["keypoints"]["y"]))
            top_left = a["box"]["x1"], a["box"]["y1"]
            width = a["box"]["x2"] - a["box"]["x1"]
            height = a["box"]["y2"] - a["box"]["y1"]

            obj = KeypointObject(
                a["name"], keypoints, (top_left, width, height), a["confidence"]
            )
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

        return DartPrediction(scores, matrix, objects)
