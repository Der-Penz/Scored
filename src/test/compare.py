from typing import List, Tuple
from src.board.dartboard import DartBoard, DartThrow
from src.perspective import compute_perspective, warp_point
from src.predictor import DartPrediction, DartPredictor
from src.preparation.dataset.yolo import YoloAnnotations
from collections import Counter


class ThrowComparison:
    """
    Class to hold the comparison of throws
    """

    def __init__(self, pred: List[DartThrow], truth: List[DartThrow]):
        self.pred = pred
        self.truth = truth
        self.cm = calculate_confusion_matrix(pred, truth)

    def confusion_matrix(self) -> Tuple[int, int, int, int]:
        """
        Get the confusion matrix of the comparison
        :return: (TP, FP, FN, TN)
        """
        return self.cm

    def correct(self) -> bool:
        """
        Check if the comparison is correct
        """
        return self.cm[0] == len(self.truth)

    def off_by(self) -> int:
        """
        Check how many throws are off.
        Positive if there are more throws in the prediction than in the ground truth.
        Negative if there are more throws in the ground truth than in the prediction.

        :return: the number of throws that are off
        """
        return len(self.missing) - len(self.additional)
    
    def num_truth_darts(self) -> int:
        """
        Get the number of darts in the ground truth
        """
        return len(self.truth)


def calculate_confusion_matrix(pred: List[DartThrow], truth: List[DartThrow]):
    """
    Compute confusion matrix for the predicted and ground truth throws.
    In this context, TN is not used, so it is set to 0.

    :param pred: the predicted throws
    :param truth: the ground truth throws

    :return: (TP, FP, FN, TN)
    """
    counter_pred = Counter(pred)
    counter_truth = Counter(truth)

    # True Positives: intersection of predicted and ground truth
    tp = sum(
        min(counter_pred[key], counter_truth[key])
        for key in counter_pred & counter_truth
    )

    # False Positives: items in predicted but not in ground truth
    fp = sum(counter_pred[key] for key in counter_pred - counter_truth)

    # False Negatives: items in ground truth but not in predicted
    fn = sum(counter_truth[key] for key in counter_truth - counter_pred)

    tn = 0  # TN is not used in this context

    return tp, fp, fn, tn


def eval_prediction(
    annotation: YoloAnnotations, prediction: DartPrediction, board: DartBoard
) -> ThrowComparison:
    """
    Evaluate the prediction against the ground truth annotation.

    :param annotation: the ground truth annotation
    :param prediction: the predicted annotation
    :param board: the dartboard to use for the evaluation

    :return: the comparison of the throws

    """

    darts = annotation.get_class("dart")
    dartboard = annotation.get_class("dartboard")[0]

    src_points = dartboard.keypoints_pos()
    ground_truth_matrix = compute_perspective(
        src_points, DartPredictor.get_perspective_transform_points(board)
    )

    truth = []
    for dart in darts:
        tip = [keypoint for keypoint in dart.keypoints if keypoint[2] == "dart"][0]
        tip = warp_point(ground_truth_matrix, tuple(tip[:2]))

        throw = board.score_dart(tip, relative=False)
        truth.append(throw)

    predicted = prediction.scores

    return ThrowComparison(predicted, truth)
