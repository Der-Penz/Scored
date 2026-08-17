import tkinter as tk
import ttkbootstrap as ttk


from gui.view.player_view import PlayerView
from gui.view.scorepad_view import ScorepadView
from gui.view.turn_view import TurnView


class DartGameView(ttk.Frame):
    """View for playing a single dart leg."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, style="Card.TFrame", padding=5)
        self.pack(fill="both", expand=True)

        self._create_widgets()

    def _create_widgets(self) -> None:
        self.player_view = PlayerView(self)
        self.player_view.pack(side="top", fill="x")

        self.turn_view = TurnView(self)
        self.turn_view.pack(side="top", fill="x")

        self.scorepad_view = ScorepadView(self)
        self.scorepad_view.pack(side="bottom", fill="both", expand=True)
