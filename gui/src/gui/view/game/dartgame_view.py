import tkinter as tk

import ttkbootstrap as ttk
from gui.view.game.player_view import PlayerView
from gui.view.game.scorepad_view import ScorepadView
from gui.view.game.turn_view import TurnView


class DartGameView(ttk.Frame):
    """View for playing a single dart leg."""

    def __init__(self, master: tk.Misc):
        # Geometry is owned by the PanelManager, so this only lays out its
        # children. Padding comes from the column this view is docked in.
        super().__init__(master, style="Card.TFrame", padding=0)

        self._create_widgets()

    def _create_widgets(self) -> None:
        self.player_view = PlayerView(self)
        self.player_view.pack(side="top", fill="x")

        self.turn_view = TurnView(self)
        self.turn_view.pack(side="top", fill="x")

        self.scorepad_view = ScorepadView(self)
        self.scorepad_view.pack(side="bottom", fill="both", expand=True)
