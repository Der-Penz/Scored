from __future__ import annotations

import tkinter as tk

import ttkbootstrap as ttk

from gui.helper import center_dialog


class TopLevelView(ttk.Toplevel):
    """
    Base class for a view that lives in a window of its own.

    Subclasses declare their window through the class attributes below;
    anything left at its default stays unset. A view marked as a dialog is
    also centered over whatever opened it, while a plain top-level window
    (such as a floating panel) is left to the window manager.
    """

    TITLE = ""
    GEOMETRY = ""
    MIN_SIZE: tuple[int, int] | None = None
    DIALOG = True

    def __init__(self, master: tk.Misc) -> None:
        super().__init__()
        if self.TITLE:
            self.title(self.TITLE)
        self.transient(master.winfo_toplevel())
        if self.GEOMETRY:
            self.geometry(self.GEOMETRY)
        if self.MIN_SIZE is not None:
            self.minsize(*self.MIN_SIZE)
        if self.DIALOG:
            center_dialog(self, master)
