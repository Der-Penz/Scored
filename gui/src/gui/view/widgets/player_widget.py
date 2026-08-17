import ttkbootstrap as ttk
from scored_lib.game.player import Player


import tkinter as tk
from tkinter import font as tkfont

ACTIVE_STYLE = "info"
INACTIVE_STYLE = "secondary"
CARD_MIN_WIDTH = 120


class PlayerWidget(ttk.Frame):
    """Card widget displaying an individual player's details and score."""

    def __init__(self, master: tk.Misc, player: Player) -> None:
        """Initialize the player widget card.

        Parameters
        ----------
        master : tk.Misc
            Parent widget.
        player : Player
            The player associated with this widget.
        """
        super().__init__(master, style="Card.TFrame", padding=(8, 4))

        # Enforce minimum width while letting height auto-calculate
        self.columnconfigure(0, minsize=CARD_MIN_WIDTH, weight=1)

        self.player = player

        self.name_label = ttk.Label(
            self,
            text=player.name,
            anchor="center",
            font=tkfont.Font(size=10, weight="bold"),
            bootstyle=INACTIVE_STYLE,
        )
        self.name_label.grid(row=0, column=0, sticky="ew")

        self.score_label = ttk.Label(
            self,
            text="--",
            anchor="center",
            padding=(4, 2),
            font=tkfont.Font(size=20, weight="bold"),
            bootstyle=INACTIVE_STYLE,
        )
        self.score_label.grid(row=1, column=0, sticky="ew")

        self.avg_label = ttk.Label(
            self,
            text="Avg: -",
            anchor="center",
            font=tkfont.Font(size=9),
            bootstyle=INACTIVE_STYLE,
        )
        self.avg_label.grid(row=2, column=0, sticky="ew")

    def update_data(self, score: int | None = None, avg: float | None = None) -> None:
        """Update the player score and average values.

        Parameters
        ----------
        score : int | None, optional
            New score.
        avg : float | None, optional
            New average.
        """
        self.score_label.config(text=str(score) if score is not None else "---")
        self.avg_label.config(text=f"Avg: {avg:.1f}" if avg is not None else "Avg: -")

    def highlight(self, highlight: bool) -> None:
        """Highlight the widget to show active turn."""
        for widget in (self.name_label, self.score_label, self.avg_label):
            widget.config(bootstyle=ACTIVE_STYLE if highlight else INACTIVE_STYLE)
