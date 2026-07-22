import math
import tkinter as tk

from gui.util import to_relative_coordinates
from scored_lib.util.angle import get_clockwise_angle


def draw_alignment_lines(canvas: tk.Canvas) -> None:
    """
    Draws radial alignment lines and angle labels for dartboard debugging.

    Draws uniform lines every 9 degrees with text labels indicating
    the degree of each segment.

    Parameters
    ----------
    canvas : tk.Canvas
        The Tkinter canvas widget to draw the lines on.
    """
    # Get canvas dimensions and calculate center point
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    cx, cy = width / 2, height / 2

    # Calculate a radius large enough to extend beyond the canvas corners
    max_radius = math.sqrt(cx**2 + cy**2)

    # Calculate a radius for text labels (90% of the smallest dimension)
    label_radius = min(cx, cy) * 0.9

    # Clear existing alignment elements before redrawing to prevent ghosting
    canvas.delete("alignment_debug")

    for degree in range(0, 360, 9):
        # Calculate the actual radial angle for this step
        current_angle = degree
        angle_rad = math.radians(current_angle)

        # Calculate the outer boundary coordinates for the line
        ex = cx + (max_radius * math.cos(angle_rad))
        ey = cy + (max_radius * math.sin(angle_rad))

        # Calculate position for the text label
        tx = cx + (label_radius * math.cos(angle_rad))
        ty = cy + (label_radius * math.sin(angle_rad))

        # Draw the 9-degree alignment line
        canvas.create_line(
            cx, cy, ex, ey, fill="gray", dash=(2, 2), width=1, tags="alignment_debug"
        )

        # Draw the angle label at the calculated radius
        canvas.create_text(
            tx,
            ty,
            text=f"{degree}°",
            fill="white",
            font=("Arial", 8, "bold"),
            tags="alignment_debug",
        )

    # Ensure the debug layer stays on top of other canvas elements
    canvas.tag_raise("alignment_debug")
