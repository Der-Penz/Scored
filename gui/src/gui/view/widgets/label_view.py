import tkinter as tk

from gui.view.widgets.board_view import BoardView


class LabelView(BoardView):
    def __init__(self, parent, name):
        super().__init__(parent, name)
        label = tk.Label(self, text=self.name, font=("Arial", 16))
        label.pack(expand=True, fill="both")  # centers the label
