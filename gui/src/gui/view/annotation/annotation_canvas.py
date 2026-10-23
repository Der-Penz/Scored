from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Sequence
from dataclasses import dataclass

import ttkbootstrap as ttk
from PIL import Image, ImageTk

POINT_RADIUS = 6
CROSS_SIZE = 6
CROSS_WIDTH = 2
DART_LINE_WIDTH = 2
DART_LINE_DASH = (4, 4)
POINT_LABEL_OFFSET = 8
POINT_COLOR = "#FFC83D"
POINT_OUTLINE = "black"
POINT_LABEL_COLOR = "white"
POINT_LABEL_FONT = ("Arial", 9, "bold")

HIDDEN_POINT_COLOR = "#8A8A8A"
HIDDEN_POINT_DASH = (2, 2)
HIDDEN_LABEL_COLOR = "#B0B0B0"

SELECTED_RADIUS = 10
SELECTED_COLOR = "#FF001E"

HIT_RADIUS = 12

IMAGE_TAG = "frame_image"
POINT_TAG = "point"

MIN_ZOOM = 1.0
MAX_ZOOM = 16.0
ZOOM_STEP = 1.25

ZOOM_BADGE_TAG = "zoom_badge"
ZOOM_BADGE_FONT = ("Arial", 10, "bold")


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
    index : int
        Index of the dart this point belongs to in the annotation, -1 for the
        point of a draft that is not attached to a dart yet.
    kind : str
        Whether the point is the ``"tip"`` or the ``"flight"`` of its dart.
    label : str
        Short text drawn next to the marker, e.g. ``"T"`` for the tip.
    color : str
        Fill color of the marker.
    selected : bool
        True when the dart this point belongs to is selected.
    visible : bool
        True when the point is visible in the image. Hidden points are drawn
        hollow and dashed in a muted color so they can still be selected and
        toggled back on.
    """

    x: float
    y: float
    index: int = -1
    kind: str = "tip"
    label: str = ""
    color: str = POINT_COLOR
    selected: bool = False
    visible: bool = True


class AnnotationCanvas(ttk.Frame):
    """Canvas that shows a camera frame with annotation points on it.

    The frame is scaled to fit the widget while keeping its aspect ratio and
    is redrawn whenever the widget changes size. Points are kept in relative
    image coordinates, so they follow the frame when it is rescaled.

    The frame is drawn letterboxed in the middle of the canvas, so the widget
    coordinates of a point depend on the current size. The view converts
    between the two spaces through :meth:`to_image` and :meth:`hit_test`.
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
        self._zoom = 1.0
        self._view_center: tuple[float, float] = (0.5, 0.5)
        self._crop: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._pan_start: tuple[float, float] | None = None
        self._pan_center: tuple[float, float] = (0.5, 0.5)

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_wheel_in)
        self.canvas.bind("<Button-5>", self._on_wheel_out)
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)

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
        self._zoom = 1.0
        self._view_center = (0.5, 0.5)
        self._redraw()

    def to_image(self, x: float, y: float, clamp: bool = False) -> tuple[float, float] | None:
        """Convert a widget coordinate into relative image space.

        Parameters
        ----------
        x, y : float
            The widget coordinate of the click, relative to the canvas.
        clamp : bool
            When True a point outside the frame is clamped to its border instead
            of being rejected, which keeps a drag inside the picture.

        Returns
        -------
        tuple[float, float] | None
            The image space coordinate between 0.0 and 1.0, or None when the
            widget position is not on the frame and *clamp* is False.
        """
        if self._image is None or self._drawn_size is None:
            return None
        if self._display_size[0] <= 0 or self._display_size[1] <= 0:
            return None

        origin_x, origin_y = self._origin
        display_width, display_height = self._display_size
        relative_x = (x - origin_x) / display_width
        relative_y = (y - origin_y) / display_height
        left, top, right, bottom = self._crop
        crop_width = max(right - left, 1)
        crop_height = max(bottom - top, 1)
        image_x = (left + relative_x * crop_width) / self._image.width
        image_y = (top + relative_y * crop_height) / self._image.height

        if clamp:
            return (
                min(1.0, max(0.0, image_x)),
                min(1.0, max(0.0, image_y)),
            )
        if 0.0 <= image_x <= 1.0 and 0.0 <= image_y <= 1.0:
            return (image_x, image_y)
        return None

    def _to_widget(self, point_x: float, point_y: float) -> tuple[float, float]:
        """Convert image space coordinates into canvas widget coordinates."""
        origin_x, origin_y = self._origin
        display_width, display_height = self._display_size
        left, top, right, bottom = self._crop
        crop_width = max(right - left, 1)
        crop_height = max(bottom - top, 1)
        image_x = point_x * self._image.width
        image_y = point_y * self._image.height
        return (
            origin_x + (image_x - left) / crop_width * display_width,
            origin_y + (image_y - top) / crop_height * display_height,
        )

    def hit_test(self, x: float, y: float) -> tuple[int, str] | None:
        """Find the labeled point closest to a widget coordinate.

        Parameters
        ----------
        x, y : float
            The widget coordinate to test, relative to the canvas.

        Returns
        -------
        tuple[int, str] | None
            The ``(index, kind)`` of the nearest point within
            :data:`HIT_RADIUS` pixels, or None when the click missed every point.
        """
        if self._image is None or self._drawn_size is None:
            return None

        best: tuple[int, str] | None = None
        best_distance = HIT_RADIUS
        for point in self._points:
            if point.index < 0:
                continue
            widget_x, widget_y = self._to_widget(point.x, point.y)
            distance = math.hypot(widget_x - x, widget_y - y)
            if distance <= best_distance:
                best_distance = distance
                best = (point.index, point.kind)
        return best

    def reset_zoom(self) -> None:
        """Zoom back out so the whole frame is visible again."""
        self._zoom = 1.0
        self._view_center = (0.5, 0.5)
        self._redraw()

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        """Zoom by *factor* around the cursor, keeping its image point put."""
        if self._image is None or self._drawn_size is None:
            return
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        anchor = self.to_image(x, y)
        if anchor is None:
            anchor = self._view_center
        self._zoom = new_zoom
        if new_zoom <= 1.0:
            self._view_center = (0.5, 0.5)
        else:
            canvas_width, canvas_height = self._drawn_size
            image_width, image_height = self._image.size
            scale = min(canvas_width / image_width, canvas_height / image_height) * new_zoom
            self._view_center = (
                anchor[0] - (x - canvas_width / 2) / scale / image_width,
                anchor[1] - (y - canvas_height / 2) / scale / image_height,
            )
        self._redraw()

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Zoom in and out around the mouse pointer."""
        factor = ZOOM_STEP ** (event.delta / 120)
        self._zoom_at(event.x, event.y, factor)

    def _on_wheel_in(self, event: tk.Event) -> None:
        """Zoom in a notch (used on platforms without a wheel delta)."""
        self._zoom_at(event.x, event.y, ZOOM_STEP)

    def _on_wheel_out(self, event: tk.Event) -> None:
        """Zoom out a notch (used on platforms without a wheel delta)."""
        self._zoom_at(event.x, event.y, 1 / ZOOM_STEP)

    def _on_pan_start(self, event: tk.Event) -> None:
        """Start panning with the middle button once the view is zoomed in."""
        if self._zoom <= 1.0:
            self._pan_start = None
            return
        self._pan_start = (event.x, event.y)
        self._pan_center = self._view_center
        self.canvas.configure(cursor="fleur")

    def _on_pan_move(self, event: tk.Event) -> None:
        """Pan the view by dragging with the middle button."""
        if self._pan_start is None or self._image is None or self._drawn_size is None:
            return
        image_width, image_height = self._image.size
        canvas_width, canvas_height = self._drawn_size
        scale = min(canvas_width / image_width, canvas_height / image_height) * self._zoom
        delta_x = event.x - self._pan_start[0]
        delta_y = event.y - self._pan_start[1]
        self._view_center = (
            self._pan_center[0] - delta_x / scale / image_width,
            self._pan_center[1] - delta_y / scale / image_height,
        )
        self._redraw()

    def _on_pan_end(self, _event: tk.Event) -> None:
        """End panning and restore the pointer."""
        self._pan_start = None
        self.canvas.configure(cursor="")

    def _on_resize(self, event: tk.Event) -> None:
        """Rescale the frame, but only when the canvas really changed size."""
        if self._drawn_size == (event.width, event.height):
            return
        self._redraw()

    def _redraw(self) -> None:
        """Fit the frame (and the zoomed view) into the current canvas size."""
        self.canvas.delete("all")
        self._photo = None
        self._drawn_size = None
        self._origin = (0.0, 0.0)
        self._display_size = (0.0, 0.0)
        self._crop = (0, 0, 0, 0)

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        self._drawn_size = (width, height)
        if self._image is None:
            return

        if self._zoom <= 1.0:
            self._view_center = (0.5, 0.5)
            crop = (0, 0, *self._image.size)
        else:
            crop = self._crop_box(width, height)
        self._crop = crop

        crop_width = max(crop[2] - crop[0], 1)
        crop_height = max(crop[3] - crop[1], 1)
        scale = min(width / crop_width, height / crop_height)
        display = (
            max(1, round(crop_width * scale)),
            max(1, round(crop_height * scale)),
        )
        self._origin = ((width - display[0]) / 2, (height - display[1]) / 2)
        self._display_size = display

        source = self._image.crop(crop)
        resized = source.resize(display, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(*self._origin, image=self._photo, anchor="nw", tag=IMAGE_TAG)
        self._draw_points()
        self._draw_zoom_badge()

    def _crop_box(self, canvas_width: int, canvas_height: int) -> tuple[int, int, int, int]:
        """Return the visible image rectangle ``(left, top, right, bottom)``."""
        image_width, image_height = self._image.size
        fit = min(canvas_width / image_width, canvas_height / image_height)
        scale = fit * self._zoom
        view_width = canvas_width / scale
        view_height = canvas_height / scale
        center_x = self._view_center[0] * image_width
        center_y = self._view_center[1] * image_height
        left = min(max(center_x - view_width / 2, 0.0), max(image_width - view_width, 0.0))
        top = min(max(center_y - view_height / 2, 0.0), max(image_height - view_height, 0.0))
        left, top = round(left), round(top)
        return (
            left,
            top,
            min(image_width, max(left + 1, round(left + view_width))),
            min(image_height, max(top + 1, round(top + view_height))),
        )

    def _draw_points(self) -> None:
        """Place the points at their position on the frame."""
        self.canvas.delete(POINT_TAG)
        if self._image is None or self._drawn_size is None:
            return

        placed: list[tuple[AnnotationPoint, float, float]] = []
        ends: dict[int, dict[str, tuple[float, float]]] = {}
        colors: dict[int, str] = {}
        visible: dict[int, dict[str, bool]] = {}
        for point in self._points:
            x, y = self._to_widget(point.x, point.y)
            placed.append((point, x, y))
            if point.index >= 0:
                ends.setdefault(point.index, {})[point.kind] = (x, y)
                colors[point.index] = point.color
                visible.setdefault(point.index, {})[point.kind] = point.visible

        self._draw_dart_lines(ends, colors, visible)

        for point, x, y in placed:
            draft = point.index < 0
            color = point.color if point.visible else HIDDEN_POINT_COLOR

            if point.kind == "tip":
                self._draw_cross(x, y, color, draft, point.visible)
            else:
                self._draw_circle(x, y, color, draft, point.visible)

            if point.selected:
                self.canvas.create_oval(
                    x - SELECTED_RADIUS,
                    y - SELECTED_RADIUS,
                    x + SELECTED_RADIUS,
                    y + SELECTED_RADIUS,
                    outline=SELECTED_COLOR,
                    width=2,
                    dash=None if point.visible else HIDDEN_POINT_DASH,
                    tag=POINT_TAG,
                )
            if point.label:
                self.canvas.create_text(
                    x + POINT_LABEL_OFFSET,
                    y - POINT_LABEL_OFFSET,
                    text=point.label,
                    fill=POINT_LABEL_COLOR if point.visible else HIDDEN_LABEL_COLOR,
                    anchor="sw",
                    font=POINT_LABEL_FONT,
                    tag=POINT_TAG,
                )

    def _draw_dart_lines(
        self,
        ends: dict[int, dict[str, tuple[float, float]]],
        colors: dict[int, str],
        visible: dict[int, dict[str, bool]],
    ) -> None:
        """Connect the tip and flight of every dart with a dotted line."""
        for index, points in ends.items():
            tip = points.get("tip")
            flight = points.get("flight")
            if tip is None or flight is None:
                continue
            flags = visible.get(index, {})
            hidden = not flags.get("tip", True) or not flags.get("flight", True)
            self.canvas.create_line(
                tip[0],
                tip[1],
                flight[0],
                flight[1],
                fill=HIDDEN_POINT_COLOR if hidden else colors[index],
                width=DART_LINE_WIDTH,
                dash=DART_LINE_DASH,
                tag=POINT_TAG,
            )

    def _draw_cross(
        self, x: float, y: float, color: str, draft: bool, visible: bool = True
    ) -> None:
        """Draw the tip of a dart as a cross, dashed when it is hidden."""
        size = CROSS_SIZE + (2 if draft else 0)
        dash = None if visible else HIDDEN_POINT_DASH
        self.canvas.create_line(
            x - size,
            y - size,
            x + size,
            y + size,
            fill=color,
            width=CROSS_WIDTH,
            capstyle="round",
            dash=dash,
            tag=POINT_TAG,
        )
        self.canvas.create_line(
            x - size,
            y + size,
            x + size,
            y - size,
            fill=color,
            width=CROSS_WIDTH,
            capstyle="round",
            dash=dash,
            tag=POINT_TAG,
        )

    def _draw_circle(
        self, x: float, y: float, color: str, draft: bool, visible: bool = True
    ) -> None:
        """Draw the flight of a dart as a filled circle, hollow when hidden."""
        radius = POINT_RADIUS + 1 if draft else POINT_RADIUS
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color if visible else "",
            outline=color if not visible else POINT_OUTLINE,
            dash=None if visible else HIDDEN_POINT_DASH,
            tag=POINT_TAG,
        )

    def _draw_zoom_badge(self) -> None:
        """Overlay the current magnification in the corner of the canvas."""
        if self._image is None or self._drawn_size is None:
            return
        width, height = self._drawn_size
        x, y = width - 8, height - 8
        text = f"{self._zoom * 100:.0f}%"
        for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            self.canvas.create_text(
                x + offset_x,
                y + offset_y,
                text=text,
                anchor="se",
                fill="black",
                font=ZOOM_BADGE_FONT,
                tag=ZOOM_BADGE_TAG,
            )
        self.canvas.create_text(
            x,
            y,
            text=text,
            anchor="se",
            fill="white",
            font=ZOOM_BADGE_FONT,
            tag=ZOOM_BADGE_TAG,
        )
