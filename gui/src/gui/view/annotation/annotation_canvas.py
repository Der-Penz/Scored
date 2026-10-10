from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from dataclasses import dataclass

import ttkbootstrap as ttk
from PIL import Image, ImageTk

POINT_RADIUS = 6
POINT_LABEL_OFFSET = 8
POINT_COLOR = "#FFC83D"
POINT_OUTLINE = "black"
POINT_LABEL_COLOR = "white"
POINT_LABEL_FONT = ("Arial", 9, "bold")

IMAGE_TAG = "frame_image"
POINT_TAG = "point"


@dataclass(frozen=True, slots=True)
class AnnotationPoint:
    """A single labeled point drawn on top of the displayed frame.

    Parameters
    ----------
    x : float
        Relative image space x coordinate (0.0-1.0), so the point stays on
        the same spot of the picture whatever size the canvas currently has.
    y : float
        Relative image space y coordinate (0.0-1.0).
    label : str
        Short text drawn next to the marker, e.g. ``"T"`` for the tip.
    color : str
        Fill color of the marker.
    """

    x: float
    y: float
    label: str = ""
    color: str = POINT_COLOR


class AnnotationCanvas(ttk.Frame):
    """Canvas that shows a camera frame with annotation points on it.

    The frame is scaled to fit the widget while keeping its aspect ratio and
    is redrawn whenever the widget changes size. Points are kept in relative
    image coordinates, so they follow the frame when it is rescaled.

    The view only draws: the controller hands over the frame and the points
    through :meth:`set_image` and :meth:`set_points`.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = ttk.Canvas(self, highlightthickness=0, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self._image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._points: list[AnnotationPoint] = []
        self._origin: tuple[float, float] = (0.0, 0.0)
        self._display_size: tuple[float, float] = (0.0, 0.0)
        self._drawn_size: tuple[int, int] | None = None

        self.canvas.bind("<Configure>", self._on_resize)

    # -- content -----------------------------------------------------------

    def set_image(self, image: Image.Image | None) -> None:
        """Display *image* scaled to fit the canvas, or clear it for ``None``."""
        self._image = image
        self._redraw()

    def set_points(self, points: Sequence[AnnotationPoint]) -> None:
        """Replace the points drawn on the frame."""
        self._points = list(points)
        self._draw_points()

    def clear_points(self) -> None:
        """Remove all points from the frame."""
        self.set_points(())

    def clear(self) -> None:
        """Remove the frame and every point drawn on it."""
        self._image = None
        self._points = []
        self._redraw()

    # -- drawing -----------------------------------------------------------

    def _on_resize(self, event: tk.Event) -> None:
        """Rescale the frame, but only when the canvas really changed size."""
        if self._drawn_size == (event.width, event.height):
            return
        self._redraw()

    def _redraw(self) -> None:
        """Fit the frame into the current canvas size and draw it."""
        self.canvas.delete("all")
        self._photo = None
        self._drawn_size = None
        self._origin = (0.0, 0.0)
        self._display_size = (0.0, 0.0)

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        self._drawn_size = (width, height)
        if self._image is None:
            return

        image_width, image_height = self._image.size
        scale = min(width / image_width, height / image_height)
        display = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )
        self._origin = ((width - display[0]) / 2, (height - display[1]) / 2)
        self._display_size = display

        resized = self._image.resize(display, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(*self._origin, image=self._photo, anchor="nw", tag=IMAGE_TAG)
        self._draw_points()

    def _draw_points(self) -> None:
        """Place the points at their position on the frame."""
        self.canvas.delete(POINT_TAG)
        if self._image is None or self._drawn_size is None:
            return

        origin_x, origin_y = self._origin
        display_width, display_height = self._display_size

        for point in self._points:
            x = origin_x + point.x * display_width
            y = origin_y + point.y * display_height

            self.canvas.create_oval(
                x - POINT_RADIUS,
                y - POINT_RADIUS,
                x + POINT_RADIUS,
                y + POINT_RADIUS,
                fill=point.color,
                outline=POINT_OUTLINE,
                tag=POINT_TAG,
            )
            if point.label:
                self.canvas.create_text(
                    x + POINT_LABEL_OFFSET,
                    y - POINT_LABEL_OFFSET,
                    text=point.label,
                    fill=POINT_LABEL_COLOR,
                    anchor="sw",
                    font=POINT_LABEL_FONT,
                    tag=POINT_TAG,
                )
