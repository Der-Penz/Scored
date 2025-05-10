from typing import Any, List, Optional, Tuple
from matplotlib import pyplot as plt

from scored.prediction.predictor import DartPrediction
from scored.util import BBYolo


def draw_bb(
    ax: Any, bb: BBYolo, label: str = None, label_kwargs : dict = dict(), **kwargs
) -> None:
    """
    Draws a bounding box on the given image.

    :param image: The image on which the bounding box will be drawn.
    :param bb: The bounding box to draw.
    :param label: The label of the bounding box.
    :param kwargs: Additional arguments for label style.
    :param kwargs: Additional arguments for plt.Rectangle
    """

    kwargs.setdefault("fill", False)
    kwargs.setdefault("edgecolor", "b")
    kwargs.setdefault("lw", 0.5)

    label_kwargs.setdefault("c", "b")

    top_left = (bb.center[0] - bb.width / 2, bb.center[1] - bb.height / 2)
    ax.add_patch(plt.Rectangle(top_left, bb.width, bb.height, **kwargs))

    if label:
        ax.text(top_left[0], top_left[1], label, **label_kwargs)

def draw_keypoints(
    ax: Any, keypoints: List[Tuple[float, float]], labels: Optional[List[str]] = None, label_kwargs : dict = dict(), **kwargs
) -> None:
    """
    Draws a list of keypoints on the given image.

    :param image: The image on which the keypoints will be drawn.
    :param keypoints: The keypoints to draw.
    :param label: The label of the keypoints.
    :param kwargs: Additional arguments for label style.
    """

    kwargs.setdefault("markersize", 2)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("c", "b")

    label_kwargs.setdefault("c", "b")

    if labels is None:
        labels = [None] * len(keypoints)

    for i, keypoint in enumerate(keypoints):
        ax.plot(keypoint[0], keypoint[1], **kwargs)
        ax.text(keypoint[0], keypoint[1], labels[i] if labels[i] is not None else i, **label_kwargs)


def draw_prediction( ax: Any, prediction: DartPrediction): 
    """
    Draws a prediction on the given image.

    :param image: The image on which the prediction will be drawn.
    :param prediction: The prediction to draw.
    """

    dart_board = prediction.get_dartboard()
    draw_bb(ax, dart_board.bb, label=f"{dart_board.name} {dart_board.conf:.2f}", color="r", label_kwargs={"color": "r"})
    draw_keypoints(ax, dart_board.keypoints, label=f"{dart_board.name} {dart_board.conf:.2f}")

    for obj in prediction.get_darts():
        draw_bb(ax, obj.bb, label=f"{obj.name} {obj.conf:.2f}")
        draw_keypoints(ax, obj.keypoints, label=f"{obj.name} {obj.conf:.2f}")

    
    ax.set_title(f"Scores: {" | ".join([throw.short_label for throw in prediction.scores])} = {prediction.sum_score()}", fontsize=16)
    ax.axis("off")
