from typing import Any, Tuple
from matplotlib import pyplot as plt


def draw_bb(
    ax: Any, bb: Tuple[Tuple[float, float], float, float], label: str = None, label_kwargs : dict = dict(), **kwargs
) -> None:
    """
    Draws a bounding box on the given image.

    :param image: The image on which the bounding box will be drawn.
    :param bb: The bounding box to draw. Format: (topleft corner, width, height)
    :param label: The label of the bounding box.
    :param kwargs: Additional arguments for label style.
    :param kwargs: Additional arguments for plt.Rectangle
    """

    kwargs.setdefault("fill", False)
    kwargs.setdefault("edgecolor", "r")
    kwargs.setdefault("lw", 0.5)

    label_kwargs.setdefault("c", "r")

    ax.add_patch(plt.Rectangle(bb[0], bb[1], bb[2], **kwargs))

    if label:
        ax.text(bb[0][0], bb[0][1], label, **label_kwargs)
