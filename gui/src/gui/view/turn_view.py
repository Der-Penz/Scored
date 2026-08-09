import tkinter as tk
from tkinter import ttk

class TurnView(ttk.Frame):
    """Turn view with dart throws"""

    def __init__(self, master: tk.Misc):
        super().__init__(master, style="Card.TFrame")

        self._create_widgets()

    def _create_widgets(self) -> None:
        pass