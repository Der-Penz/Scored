from src.board.dartboard import DartBoard
from src.perspective import compute_perspective, warp_point
from src.predictor import DartPrediction, DartPredictor
from src.preparation.dataset.yolo import YoloAnnotations


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
