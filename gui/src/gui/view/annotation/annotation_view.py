from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence

import numpy as np
import ttkbootstrap as ttk
from PIL import Image

from gui.view.annotation.annotation_canvas import AnnotationCanvas

THROWS = (1, 2, 3)

INDEX_COLUMN = {"text": "Index", "anchor": "center", "width": 70, "stretch": False}
TIP_COLUMN = {"text": "Tip", "anchor": "center", "stretch": True}
FLIGHT_COLUMN = {"text": "Flight", "anchor": "center", "stretch": True}

VISIBLE_ROWS = 4


class AnnotationView(ttk.Frame):
    """Panel for labeling the keypoints of the darts in a camera frame.

    """

    on_throw_selected: Callable[[int], None] = lambda _: None
    on_copy_clicked: Callable[[], None] = lambda : None
    on_delete_clicked: Callable[[], None] = lambda : None

    def __init__(
        self,
        master: tk.Misc,
    ) -> None:
        super().__init__(master, style="Card.TFrame", padding=4)
        self.create_widgets()

    def create_widgets(self) -> None:
        """Build the toolbar, the canvas and the table below it."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._create_toolbar().grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.canvas = AnnotationCanvas(self)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(0, 4))

        self._create_table().grid(row=2, column=0, sticky="ew")

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
        self._notebook.bind("<<NotebookTabChanged>>", lambda _: self.on_throw_selected)
        self._notebook.grid(row=0, column=0, sticky="w")

        ttk.Button(
            toolbar,
            text="Delete",
            bootstyle="danger-outline",
            command=lambda _: self.on_delete_clicked,
        ).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(
            toolbar, text="Copy", bootstyle="secondary", command=lambda _: self.on_copy_clicked
        ).grid(row=0, column=1)
        return toolbar
    
    def set_throw_tab(self, throw: int, enabled: bool, img: np.ndarray) -> None:
        """Add a tab for *throw* to the notebook."""
        if throw not in THROWS:
            raise ValueError(f"Invalid throw {throw}. Must be one of {THROWS}.")
        tab = self._throw_tabs[throw - 1]
        
        self._notebook.tab(tab, state="normal" if enabled else "disabled")
        
        if enabled:
            self._notebook.select(tab)
            self.canvas.set_image(Image.fromarray(img))
        

    def _create_table(self) -> ttk.Tableview:
        """Build the table listing the points of the current frame."""
        colors = ttk.Style().colors
        self._table = ttk.Tableview(
            master=self,
            coldata=[INDEX_COLUMN, TIP_COLUMN, FLIGHT_COLUMN],
            rowdata=[],
            height=VISIBLE_ROWS,
            autofit=True,
            stripecolor=(colors.light, colors.bg),  # type: ignore
        )
        return self._table

    def set_rows(self, rows: Sequence[tuple[str, str, str]]) -> None:
        """Replace the table with *rows* of ``(index, tip, flight)`` values."""
        self._table.delete_rows()
        if rows:
            self._table.insert_rows("end", [list(row) for row in rows])

    def clear(self) -> None:
        """Remove the frame, its points and the table rows."""
        self.canvas.clear()
        self.set_rows([])
