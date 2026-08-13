import tkinter as tk
import ttkbootstrap as ttk

from gui.view.dartboard_view import DartboardView
from gui.view.source_view import SourceView


class LeftView(ttk.Frame):
    """Container view for the left-side panes."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, style="Card.TFrame", padding=5)
        self._dartboard_visible = True
        self._source_visible = True

        self.dartboard_frame = ttk.Frame(self)
        self.dartboard_view = DartboardView(self.dartboard_frame)
        self.dartboard_view.pack(fill="both", expand=True)

        self.source_view = SourceView(self)
        self._refresh_panes()

    def _refresh_panes(self) -> None:
        """Rebuild the left-side stack so only visible panes are packed."""
        for frame in (self.dartboard_frame, self.source_view):
            frame.pack_forget()

        if self._dartboard_visible:
            self.dartboard_frame.pack(side="top", fill="both", expand=True)

        if self._source_visible:
            self.source_view.pack(side="top", fill="both", expand=True)

    def set_dartboard_visible(self, visible: bool) -> None:
        self._dartboard_visible = visible
        self._refresh_panes()

    def set_source_visible(self, visible: bool) -> None:
        self._source_visible = visible
        self._refresh_panes()

    def is_visible(self) -> bool:
        return self._dartboard_visible or self._source_visible
