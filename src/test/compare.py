from typing import List
from src.board.dartboard import DartBoard, DartThrow
from src.perspective import compute_perspective, warp_point
from src.predictor import DartPrediction, DartPredictor
from src.preparation.dataset.yolo import YoloAnnotations
from collections import Counter

def compare_scores(pred: List[DartThrow], truth: List[DartThrow]):
    """
    Compare two lists of throws

    :param pred: the predicted throws
    :param truth: the ground truth throws

    :return: missing throws in the prediction and throws that are not in the ground truth
    """
    counter_pred = Counter(pred)
    counter_truth = Counter(truth)

    missing = counter_truth - counter_pred
    to_much = counter_pred - counter_truth
    missing = [key for key, value in missing.items() for _ in range(value)]
    to_much = [key for key, value in to_much.items() for _ in range(value)]

    return missing, to_much


def eval_prediction(
    annotation: YoloAnnotations, prediction: DartPrediction, board: DartBoard
) -> int:
    """
    Evaluate the prediction against the ground truth annotation.
    Returns the number of correct darts and the number of false positives.
    """

    darts = annotation.get_class("dart")
    dartboard = annotation.get_class("dartboard")[0]

    src_points = dartboard.keypoints_pos()
    ground_truth_matrix = compute_perspective(
        src_points, DartPredictor.get_perspective_transform_points(board)
    )

    truth_scores = []
    for dart in darts:
        tip = [keypoint for keypoint in dart.keypoints if keypoint[2] == "dart"][0]
        tip = warp_point(ground_truth_matrix, tuple(tip[:2]))

        throw = board.score_dart(tip, relative=False)
        truth_scores.append(throw.score())

    predicted_scores = [throw.score() for throw in prediction.scores]

    predicted_scores.sort()
    truth_scores.sort()

    return (
        all(x == y for x, y in zip(predicted_scores, truth_scores)),
        predicted_scores,
        truth_scores,
    )
