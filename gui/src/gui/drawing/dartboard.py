import math

from scored_lib.dart.constants import DARTBOARD_NUMBERS, RING_DIMENSIONS, SLICE_ANGLE_DEGREES
import tkinter as tk


ODD_COLOR = "#F9DFBC"
EVEN_COLOR = "black"
ODD_COLOR_MULTIPLIER = "#D20E15"
EVEN_COLOR_MULTIPLIER = "#0D7643"


def draw_circle(
    canvas: tk.Canvas, center_x: float, center_y: float, radius: float, color: str
) -> None:
    """
    Helper function to draw a filled circle on the canvas.

    Parameters
    ----------
    canvas : tk.Canvas
        The Tkinter canvas widget to draw on.
    center_x : float
        The x-coordinate of the center of the circle.
    center_y : float
        The y-coordinate of the center of the circle.
    radius : float
        The radius of the circle.
    color : str
        The fill color for the circle.
    """
    canvas.create_oval(
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
        fill=color,
        outline="silver",
    )


def draw_arc(
    canvas: tk.Canvas,
    center_x: float,
    center_y: float,
    radius: float,
    start_angle: float,
    extent: float,
    color: str,
) -> None:
    """
    Helper function to draw an arc segment on the canvas.

    Parameters
    ----------
    canvas : tk.Canvas
        The Tkinter canvas widget to draw on.
    center_x : float
        The x-coordinate of the center of the arc.
    center_y : float
        The y-coordinate of the center of the arc.
    radius : float
        The radius of the arc.
    start_angle : float
        The starting angle of the arc in degrees.
    extent : float
        The extent of the arc in degrees.
    color : str
        The fill color for the arc segment.
    """
    canvas.create_arc(
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
        start=start_angle,
        extent=extent,
        fill=color,
        outline="silver",
    )


def draw_dartboard(canvas: tk.Canvas) -> None:
    """
    Draws a dartboard with correct segment fields and colors.

    Parameters
    ----------
    canvas : tk.Canvas
        The Tkinter canvas widget to draw the dartboard on.
    """

    width = canvas.winfo_width()
    height = canvas.winfo_height()

    scale = min(width, height)
    center_x = width / 2
    center_y = height / 2

    r_do = RING_DIMENSIONS["double_outer"] * scale / 2
    r_di = RING_DIMENSIONS["double_inner"] * scale / 2
    r_to = RING_DIMENSIONS["triple_outer"] * scale / 2
    r_ti = RING_DIMENSIONS["triple_inner"] * scale / 2
    r_bo = RING_DIMENSIONS["outer_bull"] * scale / 2
    r_bi = RING_DIMENSIONS["inner_bull"] * scale / 2

    r_text = r_do * 1.1
    r_background = r_do * 1.2
    draw_circle(canvas, center_x, center_y, r_background, "black")  # Background

    offset = 4  # Offset to align with the defined numbers since degrees start at number 6 but the numbers at 1
    for i, number in enumerate(DARTBOARD_NUMBERS):
        # Calculate the center of the segment
        center_angle_logic = (i - offset) * SLICE_ANGLE_DEGREES
        center_angle_trig = -center_angle_logic

        # center angle by adding half the slice angle
        arc_start = center_angle_trig - (SLICE_ANGLE_DEGREES / 2)

        # * Text placement uses the center angle
        text_rotation = center_angle_trig - 90
        angle_rad = math.radians(center_angle_trig)
        tx = center_x + (r_text * math.cos(angle_rad))
        ty = center_y - (r_text * math.sin(angle_rad))

        canvas.create_text(
            tx,
            ty,
            text=str(number),
            fill="white",
            angle=text_rotation,
            font=("Arial", int(scale * 0.03), "bold"),
        )

        if i % 2 == 0:
            color_single = EVEN_COLOR
            color_double_triple = EVEN_COLOR_MULTIPLIER
        else:
            color_single = ODD_COLOR
            color_double_triple = ODD_COLOR_MULTIPLIER

        draw_arc(
            canvas, center_x, center_y, r_do, arc_start, SLICE_ANGLE_DEGREES, color_double_triple
        )
        draw_arc(canvas, center_x, center_y, r_di, arc_start, SLICE_ANGLE_DEGREES, color_single)
        draw_arc(
            canvas, center_x, center_y, r_to, arc_start, SLICE_ANGLE_DEGREES, color_double_triple
        )
        draw_arc(canvas, center_x, center_y, r_ti, arc_start, SLICE_ANGLE_DEGREES, color_single)

    draw_circle(canvas, center_x, center_y, r_bo, EVEN_COLOR_MULTIPLIER)  # Outer bulls
    draw_circle(canvas, center_x, center_y, r_bi, ODD_COLOR_MULTIPLIER)  # Inner bulls
