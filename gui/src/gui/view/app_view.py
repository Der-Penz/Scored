import tkinter as tk
import ttkbootstrap as ttk

from gui.view.dartgame_view import DartGameView
from gui.view.left_view import LeftView


class AppView(ttk.Frame):
    """
    The main application view that contains all other views.
    """

    def __init__(self, master: tk.Tk):
        super().__init__(master, style="Card.TFrame", padding=0)
        self.master = master

        self.create_widgets()

    def create_widgets(self):
        self.menu_frame = ttk.Frame(self.master)
        self.menu_frame.pack(fill="x", side="top")

        self.menu_separator = ttk.Separator(self.master, orient="horizontal")
        self.menu_separator.pack(fill="x", side="top")

        self.pack(fill="both", expand=True)

        self.main_paned_window = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill="both", expand=True)

        self.left_view = LeftView(self.main_paned_window)

        self._pane_ratio = 0.5

        self.game_view = DartGameView(self.main_paned_window)

        self.main_paned_window.bind("<ButtonRelease-1>", self._store_pane_ratio)
        self.main_paned_window.bind("<Configure>", self._on_paned_window_configure)
        self.refresh_left_panes()

    def refresh_left_panes(self) -> None:
        """Refresh the left-side pane layout after visibility changes."""
        current_panes = self.main_paned_window.panes()
        left_in_panes = str(self.left_view) in current_panes
        game_in_panes = str(self.game_view) in current_panes
        left_visible = self.left_view.is_visible()

        if left_visible:
            if not left_in_panes:
                if game_in_panes:
                    self.main_paned_window.forget(self.game_view)
                    self.main_paned_window.add(self.left_view)
                    self.main_paned_window.add(self.game_view)
                else:
                    self.main_paned_window.add(self.left_view)
            self.after_idle(self._apply_pane_ratio)
        elif left_in_panes:
            self.main_paned_window.forget(self.left_view)

        if not game_in_panes:
            self.main_paned_window.add(self.game_view)

    def _on_paned_window_configure(self, _event: tk.Event) -> None:
        if self.left_view.is_visible():
            self.after_idle(self._apply_pane_ratio)

    def _apply_pane_ratio(self) -> None:
        total_width = self.main_paned_window.winfo_width()
        if total_width <= 1:
            return

        sash = int(total_width * self._pane_ratio)
        self.main_paned_window.sashpos(0, sash)

    def _store_pane_ratio(self, _event: tk.Event | None = None) -> None:
        total_width = self.main_paned_window.winfo_width()
        if total_width <= 1:
            return

        try:
            sash_x, _sash_y = self.main_paned_window.sash_coord(0)
        except tk.TclError:
            return

        self._pane_ratio = max(0.0, min(1.0, sash_x / total_width))
