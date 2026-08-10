import tkinter as tk
import ttkbootstrap as ttk

from gui.view.dartgame_view import DartGameView
from gui.view.left_view import LeftView


class AppView(ttk.Frame):
    """
    The main application view that contains all other views.
    """

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.main_paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill="both", expand=True)

        self.left_view = LeftView(self.main_paned_window)

        self._pane_ratio = 0.5

        self.game_view = DartGameView(self.main_paned_window)

        self.main_paned_window.bind("<ButtonRelease-1>", self._store_pane_ratio)
        self.main_paned_window.bind("<Configure>", self._on_paned_window_configure)
        self.refresh_left_panes()

    def refresh_left_panes(self) -> None:
        """Refresh the left-side pane layout after visibility changes."""
        if self.left_view.winfo_manager():
            self.main_paned_window.forget(self.left_view)
        if self.game_view.winfo_manager():
            self.main_paned_window.forget(self.game_view)

        left_visible = self.left_view.is_visible()
        if left_visible:
            self.main_paned_window.add(self.left_view)
        self.main_paned_window.add(self.game_view)

        if not left_visible:
            self.update_idletasks()
            return

        self.after_idle(self._apply_pane_ratio)

    def _on_paned_window_configure(self, _event: tk.Event) -> None:
        if self.left_view.is_visible():
            self.after_idle(self._apply_pane_ratio)

    def _apply_pane_ratio(self) -> None:
        total_width = self.main_paned_window.winfo_width()
        if total_width <= 1:
            return

        sash = int(total_width * self._pane_ratio)
        self.main_paned_window.sash_place(0, sash, 0)

    def _store_pane_ratio(self, _event: tk.Event | None = None) -> None:
        total_width = self.main_paned_window.winfo_width()
        if total_width <= 1:
            return

        try:
            sash_x, _sash_y = self.main_paned_window.sash_coord(0)
        except tk.TclError:
            return

        self._pane_ratio = max(0.0, min(1.0, sash_x / total_width))
