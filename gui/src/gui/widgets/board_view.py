import tkinter as tk
from abc import ABC


class BoardView(tk.Frame, ABC):
    def __init__(self, parent, name):
        super().__init__(parent)
        self.name = name
