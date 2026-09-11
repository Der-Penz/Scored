import tkinter as tk
from typing import Callable, Sequence

from scored_lib.dart.dart_throw import DartThrow
from scored_lib.util.position import (
    get_segment_default_position,
    relative_to_canvas_position,
)
import ttkbootstrap as ttk

from gui.protocols.dartboard_protocol import DartboardProtocol
from gui.view.drawing.dartboard import (
    DART_COLORS,
    draw_dart_marker,
    draw_dartboard,
)
from scored_lib.dart.constants import Position
from scored_lib.game.dart_leg import ThrowResult

DART_TAG = "dart_marker"
BOTTOM_Y_NORMALIZED = 0.92
DART_RADIUS = 10

DragReleaseCallback = Callable[[int, Position], None]


class DartboardView(tk.Frame, DartboardProtocol):
    """Canvas-backed view for drawing the digital dartboard."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, bg="black")
        self.canvas = ttk.Canvas(self, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self._on_resize: Callable[[], None] | None = None
        self._on_drag_release: DragReleaseCallback | None = None

        self._board_size: float = 0.0
        self._board_ox: float = 0.0
        self._board_oy: float = 0.0

        self._dragged: tuple[int, str] | None = None
        self._drag_last: tuple[float, float] = (0.0, 0.0)

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.after_idle(self.redraw)

    def set_resize_callback(self, callback: Callable[[], None]) -> None:
        self._on_resize = callback

    def set_drag_release_callback(self, callback: DragReleaseCallback) -> None:
        self._on_drag_release = callback

    def _on_canvas_resize(self, event: tk.Event) -> None:
        # Keep the drawing area square so the board fills the full canvas.
        if event.width > 1 and self.canvas.winfo_height() != event.width:
            self.canvas.configure(height=event.width)
        self.redraw()
        if self._on_resize is not None:
            self._on_resize()

    def redraw(self) -> None:
        """Redraw the dartboard background using the current canvas size."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        size = min(width, height) - 2

        if size <= 1:
            return

        self.canvas.delete("all")
        draw_dartboard(self.canvas, size)

        self._board_size = size
        self._board_ox = (width - size) / 2
        self._board_oy = (height - size) / 2

    def draw_darts(self, throws: Sequence[ThrowResult]) -> None:
        """Clear and draw the given dart throws as draggable markers."""
        self.canvas.delete(DART_TAG)

        for index, throw_result in enumerate(throws):
            color = DART_COLORS[index % len(DART_COLORS)]
            label = str(throw_result.throw)

            pos = self._marker_pixel_position(
                throw_result.dart_throw, index, len(throws)
            )
            px, py = self._normalized_to_pixel(pos)
            draw_dart_marker(
                self.canvas,
                px,
                py,
                DART_RADIUS,
                color,
                label,
                tag=(DART_TAG, f"{DART_TAG}_{index}"),
            )

    def _marker_pixel_position(
        self, dart_throw: DartThrow, index: int, count: int
    ) -> Position:
        """Return the pixel position for a marker based on its throw's position."""
        position = dart_throw.position
        if position is not None:
            return relative_to_canvas_position(position)

        if (
            dart_throw.is_miss
        ):  # place missed darts evenly along the bottom of the board
            spacing = 1.0 / (count + 1)
            x = spacing * (index + 1)
            return relative_to_canvas_position((x, BOTTOM_Y_NORMALIZED))

        pos = get_segment_default_position(dart_throw.number, dart_throw.multiplier)
        return relative_to_canvas_position(pos)

    def _normalized_to_pixel(self, position: Position) -> Position:
        """Map normalized (0-1) coordinates to canvas pixel coordinates."""
        return (
            self._board_ox + position[0] * self._board_size,
            self._board_oy + position[1] * self._board_size,
        )

    def _pixel_to_normalized(self, position: Position) -> Position:
        """Map canvas pixel coordinates to normalized (0-1) coordinates."""
        if self._board_size <= 1:
            return 0.0, 0.0
        return (
            (position[0] - self._board_ox) / self._board_size,
            (position[1] - self._board_oy) / self._board_size,
        )

    def _marker_at(self, event: tk.Event) -> tuple[int, str] | None:
        """Find the index of the dart marker under the pointer, if any."""
        items = self.canvas.find_withtag(DART_TAG)
        closest = self.canvas.find_closest(event.x, event.y)
        if closest and closest[0] in items:
            for tag in self.canvas.gettags(closest[0]):
                if tag.startswith(DART_TAG + "_"):
                    index = int(tag.split("_")[-1])
                    return index, tag
        return None

    def _on_press(self, event: tk.Event) -> None:
        self._dragged = self._marker_at(event)
        if self._dragged is not None:
            self._drag_last = (event.x, event.y)

    def _on_motion(self, event: tk.Event) -> None:
        if self._dragged is None:
            return
        _, tag = self._dragged
        last_x, last_y = self._drag_last
        self.canvas.move(tag, event.x - last_x, event.y - last_y)
        self._drag_last = (event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self._dragged is None:
            return
        index, _ = self._dragged
        self._dragged = None

        position = self._pixel_to_normalized((event.x, event.y))

        if self._on_drag_release is not None:
            self._on_drag_release(index, position)

    def clear(self) -> None:
        """Clear all drawn darts and redraw the empty board."""
        self.canvas.delete(DART_TAG)
        self.redraw()
