from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import ttkbootstrap as ttk
from PIL import Image, ImageDraw, ImageTk

from gui.view.annotation.annotation_canvas import AnnotationCanvas, AnnotationPoint
from gui.view.drawing.dartboard import DART_COLORS

THROWS = (1, 2, 3)

INDEX_COLUMN = {"text": "Dart", "anchor": "center", "width": 60, "stretch": False}
SCORE_COLUMN = {"text": "Score", "anchor": "center", "width": 70, "stretch": False}
TIP_COLUMN = {"text": "Tip", "anchor": "center", "stretch": True}
FLIGHT_COLUMN = {"text": "Flight", "anchor": "center", "stretch": True}
SOURCE_COLUMN = {"text": "Source", "anchor": "center", "width": 80, "stretch": False}
VISIBLE_COLUMN = {"text": "Visible", "anchor": "center", "width": 100, "stretch": False}

VISIBLE_ROWS = 4

DRAG_THRESHOLD = 4


def _noop(*_args: object) -> None:
    """Default callback that ignores its arguments."""


class AnnotationView(ttk.Frame):
    """Panel for labeling the keypoints of the darts in a camera frame.

    A toolbar on top selects which throw is annotated (1, 2 or 3) and offers
    copy and delete actions, the canvas in the middle shows the frame and the
    points drawn on it, and the table below lists those points row by row.

    The view handles the mouse: a click on empty frame space is forwarded as
    ``on_point_placed``, a click on an existing point or a table row as
    ``on_point_selected``, and dragging an existing point as ``on_point_moved``.
    Every interaction goes through the ``on_*`` callbacks, which are plain
    attributes the controller sets again after a ``PanelRebound`` event.
    """

    on_throw_selected: Callable[[int], None] = _noop
    on_copy_clicked: Callable[[], None] = _noop
    on_delete_clicked: Callable[[], None] = _noop
    on_point_placed: Callable[[float, float], None] = _noop
    on_point_selected: Callable[[int, str | None], None] = _noop
    on_point_moved: Callable[[int, str, float, float], None] = _noop

    def __init__(
        self,
        master: tk.Misc,
    ) -> None:
        super().__init__(master, style="Card.TFrame", padding=4)
        self._announced_throw: int | None = None
        self._draft_active = False
        self._drag: tuple[int, str, float, float] | None = None
        self._selected_row: int | None = None
        self._row_iids: dict[int, str] = {}
        self._iid_to_row: dict[str, int] = {}
        self._circle_images: dict[str, ImageTk.PhotoImage] = {}
        self._last_image: np.ndarray | None = None
        self.create_widgets()

    def create_widgets(self) -> None:
        """Build the data banner, toolbar, canvas and table."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self._data_banner = ttk.Label(self, anchor="w", bootstyle="secondary")
        self._data_banner.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._create_toolbar().grid(row=1, column=0, sticky="ew", pady=(0, 4))

        self._meta_label = ttk.Label(self, style="TLabel", anchor="w")
        self._meta_label.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        self.canvas = AnnotationCanvas(self)
        self.canvas.grid(row=3, column=0, sticky="nsew", pady=(0, 4))
        self.canvas.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self._create_table().grid(row=4, column=0, sticky="ew")

        self._status_label = ttk.Label(self, style="TLabel", anchor="w")
        self._status_label.grid(row=5, column=0, sticky="ew", pady=(4, 0))

    def _create_toolbar(self) -> ttk.Frame:
        toolbar = ttk.Frame(self)
        toolbar.columnconfigure(0, weight=1)

        self._notebook = ttk.Notebook(toolbar)
        self._throw_tabs: list[ttk.Frame] = []
        for throw in THROWS:
            tab = ttk.Frame(self._notebook)
            self._notebook.add(tab, text=f"Throw {throw}")
            self._throw_tabs.append(tab)
            self._notebook.tab(tab, state="disabled")
        self._notebook.bind("<<NotebookTabChanged>>", self._on_throw_selected)
        self._notebook.grid(row=0, column=0, sticky="w")

        ttk.Button(
            toolbar,
            text="Copy Previous",
            bootstyle="secondary",
            command=self._handle_copy,
        ).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(
            toolbar,
            text="Delete",
            bootstyle="danger-outline",
            command=self._handle_delete,
        ).grid(row=0, column=2)
        return toolbar

    @property
    def selected_throw(self) -> int | None:
        """The throw whose tab is selected, or ``None`` while no throw is enabled."""
        selected = self._notebook.select()
        if not selected:
            return None
        return THROWS[self._notebook.index(selected)]

    def _on_throw_selected(self, _event: tk.Event | None = None) -> None:
        """Notify the controller when the user switches to another throw."""
        throw = self.selected_throw
        if throw is None or throw == self._announced_throw:
            return
        self._announced_throw = throw
        self.on_throw_selected(throw)

    def _handle_copy(self) -> None:
        self.on_copy_clicked()

    def _handle_delete(self) -> None:
        self.on_delete_clicked()

    def set_throw_tabs(self, available: Sequence[int]) -> None:
        """Enable the tabs the current round has a sample for."""
        for throw in THROWS:
            state = "normal" if throw in available else "disabled"
            self._notebook.tab(self._throw_tabs[throw - 1], state=state)

    def show_throw(
        self,
        throw: int,
        image: np.ndarray | None,
        points: Sequence[AnnotationPoint],
        rows: Sequence[tuple[str, str, str, str, str, str]],
        selected: int | None,
        metrics: str = "",
    ) -> None:
        """Show the frame, points and dart list of one throw of the round."""
        self._announced_throw = throw
        tab = self._throw_tabs[throw - 1]
        self._notebook.tab(tab, state="normal")
        if self.selected_throw != throw:
            self._notebook.select(tab)
        if image is not None and image is not self._last_image:
            self.canvas.set_image(Image.fromarray(image))
            self._last_image = image
        self.set_points(points)
        self.set_rows(rows)
        self._select_row(selected)
        self._meta_label.config(text=metrics)

    def set_points(self, points: Sequence[AnnotationPoint]) -> None:
        """Draw *points* on the frame; a draft point keeps clicks placing."""
        self._draft_active = any(point.index == -1 for point in points)
        self.canvas.set_points(points)

    def set_rows(self, rows: Sequence[tuple[str, str, str, str, str, str]]) -> None:
        """Replace the table with ``(index, score, tip, flight, source, visible)`` rows."""
        self._row_iids = {}
        self._iid_to_row = {}
        self._table.delete_rows()
        if rows:
            # Tableview.insert_rows inserts in reverse order, so pass the rows
            # reversed to keep dart 1 on top. It may also resort on header
            # clicks, so map each iid back to its dart by the index in the
            # first column instead of relying on the display order.
            self._table.insert_rows("end", [list(row) for row in reversed(rows)])
            for iid in self._table.view.get_children():
                values = self._table.view.item(iid, "values")
                try:
                    index = int(values[0]) - 1
                except (ValueError, TypeError):
                    continue
                if index < 0:
                    continue
                self._row_iids[index] = iid
                self._iid_to_row[iid] = index
                self._set_dart_color(iid, index)

    def _set_dart_color(self, iid: str, index: int) -> None:
        """Show a circle in the color column in the dart's palette color."""
        color = DART_COLORS[index % len(DART_COLORS)]
        self._table.view.item(iid, image=self._circle_image(color))

    def _circle_image(self, color: str) -> ImageTk.PhotoImage:
        """Return a small filled circle in *color*, cached per palette color."""
        image = self._circle_images.get(color)
        if image is None:
            size = 16
            circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            ImageDraw.Draw(circle).ellipse((1, 1, size - 2, size - 2), fill=color)
            image = ImageTk.PhotoImage(circle)
            self._circle_images[color] = image
        return image

    def set_status(self, message: str) -> None:
        """Show a short transient message under the table."""
        self._status_label.config(text=message)

    def set_data_collection(self, enabled: bool, directory: Path | None = None) -> None:
        """Show in the banner whether the labels of this panel are being saved."""
        if enabled:
            message = f"Saving data to {directory}" if directory else "Data collection is on."
            self._data_banner.configure(text=message, bootstyle="success")
        else:
            self._data_banner.configure(
                text="Data collection is off — labels are not saved.", bootstyle="warning"
            )

    def clear(self) -> None:
        """Remove the frame, its points, the table rows and the labels."""
        self.set_throw_tabs([])
        self.canvas.clear()
        self.set_points(())
        self.set_rows(())
        self._meta_label.config(text="")
        self.set_status("")
        self._selected_row = None
        self._announced_throw = None
        self._draft_active = False
        self._drag = None
        self._last_image = None

    def _on_canvas_press(self, event: tk.Event) -> None:
        """Place a point, or pick up an existing point for selection or drag."""
        self.canvas.canvas.focus_set()
        if self._draft_active:
            point = self.canvas.to_image(event.x, event.y)
            if point is not None:
                self.on_point_placed(*point)
            return

        hit = self.canvas.hit_test(event.x, event.y)
        if hit is not None:
            index, kind = hit
            self._drag = (index, kind, event.x, event.y)
            self.on_point_selected(index, kind)
        else:
            point = self.canvas.to_image(event.x, event.y)
            if point is not None:
                self.on_point_placed(*point)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        """Move the picked point once the drag leaves its click threshold."""
        if self._drag is None:
            return
        index, kind, press_x, press_y = self._drag
        if (event.x - press_x) ** 2 + (event.y - press_y) ** 2 < DRAG_THRESHOLD**2:
            return
        point = self.canvas.to_image(event.x, event.y, clamp=True)
        if point is not None:
            self.on_point_moved(index, kind, *point)

    def _on_canvas_release(self, _event: tk.Event) -> None:
        """End the drag."""
        self._drag = None

    def _create_table(self) -> ttk.Tableview:
        """Build the table listing the points of the current frame."""
        colors = ttk.Style().colors
        table = ttk.Tableview(
            master=self,
            coldata=[
                INDEX_COLUMN,
                SCORE_COLUMN,
                TIP_COLUMN,
                FLIGHT_COLUMN,
                SOURCE_COLUMN,
                VISIBLE_COLUMN,
            ],
            rowdata=[],
            height=VISIBLE_ROWS,
            autofit=True,
            stripecolor=(colors.light, colors.bg),  # type: ignore
            disable_right_click=True,
        )
        view = table.view
        # Prepend a narrow tree column that renders each dart's color circle.
        # Treeview only draws item images in the tree column, but Tableview
        # normally hides it (show="headings"), so bring it back as "Color".
        view.configure(show="tree headings")
        view.heading("#0", text="Color", anchor="center")
        view.column("#0", width=52, minwidth=40, stretch=False, anchor="center")
        view.bind("<<TreeviewSelect>>", self._on_row_select)
        self._table = table
        return table

    def _select_row(self, index: int | None) -> None:
        """Select the table row of a dart, or clear the selection for ``None``."""
        self._selected_row = index
        tree = self._table.view
        if index is None:
            if tree.selection():
                tree.selection_remove(tree.selection())
            return
        iid = self._row_iids.get(index)
        if iid is not None and iid in tree.get_children():
            tree.selection_set(iid)

    def _on_row_select(self, _event: tk.Event) -> None:
        """Notify the controller when a table row is picked by the user."""
        selection = self._table.view.selection()
        if not selection:
            return
        iid = selection[0]
        index = self._iid_to_row.get(iid)
        if index is None or index == self._selected_row:
            return
        self._selected_row = index
        self.on_point_selected(index, None)
