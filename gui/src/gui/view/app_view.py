import tkinter as tk
from PIL import Image

from gui.view.camera_feed_view import CameraFeedView
from gui.view.dartboard_view import DartboardView


class AppView(tk.Frame):
    """
    The main application view that contains all other views.
    """

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.main_paned_window = tk.PanedWindow(
            self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6
        )
        self.main_paned_window.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(self.main_paned_window, bg="black")

        self._dartboard_visible = True
        self._image_visible = True
        self._pane_ratio = 0.5

        self.dartboard_frame = tk.Frame(self.left_frame, bg="black")
        self.dartboard_view = DartboardView(self.dartboard_frame)
        self.dartboard_view.pack(fill="both", expand=True)

        self.image_frame = CameraFeedView(self.left_frame)

        self.right_frame = tk.Frame(self.main_paned_window, bg="red")

        self.main_paned_window.bind("<ButtonRelease-1>", self._store_pane_ratio)
        self.main_paned_window.bind("<Configure>", self._on_paned_window_configure)
        self._refresh_left_panes()
        self._layout_main_panes()

    def _refresh_left_panes(self) -> None:
        """Rebuild the left-side stack so only visible panes are packed."""
        for frame in (self.dartboard_frame, self.image_frame):
            frame.pack_forget()

        if self._dartboard_visible:
            self.dartboard_frame.pack(side="top", fill="both", expand=True)

        if self._image_visible:
            self.image_frame.pack(side="top", fill="both", expand=True)

        self._layout_main_panes()

    def _layout_main_panes(self) -> None:
        """Place the outer panes while preserving the current split ratio."""
        if self.left_frame.winfo_manager():
            self.main_paned_window.forget(self.left_frame)
        if self.right_frame.winfo_manager():
            self.main_paned_window.forget(self.right_frame)

        left_visible = self._dartboard_visible or self._image_visible
        if left_visible:
            self.main_paned_window.add(self.left_frame)
        self.main_paned_window.add(self.right_frame)

        if not left_visible:
            self.update_idletasks()
            return

        self.after_idle(self._apply_pane_ratio)

    def _on_paned_window_configure(self, _event: tk.Event) -> None:
        if self._dartboard_visible or self._image_visible:
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

    def set_dartboard_visible(self, visible: bool) -> None:
        self._dartboard_visible = visible
        self._refresh_left_panes()

    def set_image_visible(self, visible: bool) -> None:
        self._image_visible = visible
        self._refresh_left_panes()

    def display_image(self, pil_image: Image.Image) -> None:
        """Display a PIL image in the left-side feed view."""
        self.image_frame.display_image(pil_image)
