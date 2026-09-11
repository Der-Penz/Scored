import tkinter as tk
from typing import Sequence

import ttkbootstrap as ttk
from PIL import Image, ImageTk

from gui.protocols.dartboard_protocol import DartboardProtocol
from scored_lib.game.dart_leg import ThrowResult


class SourceView(ttk.Frame, DartboardProtocol):
    """View for the live camera or source feed."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.image_label = ttk.Label(self)
        self.image_label.pack(fill="both", expand=True)
        self._photo_image = None

    def draw_darts(self, throws: Sequence[ThrowResult]) -> None:
        pass

    def clear(self) -> None:
        pass

    def display_image(self, pil_image: Image.Image) -> None:
        """Display a PIL image in the feed view."""
        if pil_image is None:
            self.image_label.config(image="")
            self._photo_image = None
            return

        width = self.master.winfo_width()
        height = self.master.winfo_height()
        if width <= 1 or height <= 1:
            return

        image = pil_image.copy()
        image.thumbnail((width, height), Image.LANCZOS)

        self._photo_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self._photo_image)
