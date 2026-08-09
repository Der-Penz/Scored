import tkinter as tk
from tkinter import ttk

class PlayerView(ttk.Frame):
    """Player view with scores"""

    def __init__(self, master: tk.Misc):
        super().__init__(master, style="Card.TFrame")

        self._create_widgets()

    def _create_widgets(self) -> None:
        pass