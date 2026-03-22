from scored_lib.dart.constants import Position


import tkinter as tk


def to_relative_coordinates(event: tk.Event) -> Position:
    """
    Converts absolute canvas coordinates to relative coordinates centered at (0,0).

    Parameters
    ----------
    event : tk.Event
        The event containing the x and y coordinates of the mouse.

    Returns
    -------
    tuple[float, float]
        A tuple containing the relative x and y coordinates normalized to the dartboard.
    """
    canvas = event.widget
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    cx, cy = width / 2, height / 2

    scale = min(width, height) * 0.9
    board_radius = scale / 2

    # relative coordinates centered at 0,0
    rel_x = (event.x - cx) / board_radius
    rel_y = (event.y - cy) / board_radius
    return rel_x, rel_y
