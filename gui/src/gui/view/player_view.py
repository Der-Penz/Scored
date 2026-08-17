import tkinter as tk

from gui.view.widgets.player_widget import PlayerWidget, CARD_MIN_WIDTH
import ttkbootstrap as ttk

from scored_lib.game.player import Player


class PlayerView(ttk.Frame):
    """Horizontal player bar showing name, score and average."""

    def __init__(self, master: tk.Misc) -> None:
        """Initialize the player view frame.

        Parameters
        ----------
        master : tk.Misc
            Parent widget.
        """
        super().__init__(master, style="Card.TFrame", padding=10)

        self._player_widgets: dict[Player, PlayerWidget] = {}
        self._separators: list[ttk.Separator] = []

        self._inner = ttk.Frame(self, style="Card.TFrame")
        self._inner.pack(side="top", fill="both", expand=True)

    def add_player(self, player: Player) -> None:
        """Add a player widget to the view.

        Parameters
        ----------
        player : Player
            The player to add.
        """
        if player in self._player_widgets or len(self._player_widgets) >= 5:
            return

        widget = PlayerWidget(self._inner, player)
        self._player_widgets[player] = widget
        self._update_layout()

    def remove_player(self, player: Player) -> None:
        """Remove a player widget from the view.

        Parameters
        ----------
        player : Player
            The player to remove.
        """
        widget = self._player_widgets.pop(player, None)
        if widget is None:
            return

        widget.destroy()
        self._update_layout()

    def update_player(
        self, player: Player, score: int | None = None, avg: float | None = None
    ) -> None:
        """Update score and/or average for a player.

        Parameters
        ----------
        player : Player
            The player to update.
        score : int | None, optional
            New score value.
        avg : float | None, optional
            New average value.
        """
        widget = self._player_widgets.get(player)
        if widget is not None:
            widget.update_data(score=score, avg=avg)

    def set_current(self, player: Player) -> None:
        """Set the currently active player and update highlights.

        Parameters
        ----------
        player : Player
            The player whose turn is active.
        """
        for p, widget in self._player_widgets.items():
            widget.highlight(p == player)

    def _update_layout(self) -> None:
        """Update the layout of player widgets and separators."""
        for sep in self._separators:
            sep.destroy()
        self._separators.clear()

        for widget in self._player_widgets.values():
            widget.pack_forget()

        for idx, widget in enumerate(self._player_widgets.values()):
            if idx > 0:
                sep = ttk.Separator(self._inner, orient="vertical")
                sep.pack(side="left", fill="y", padx=4, pady=2)
                self._separators.append(sep)

            widget.pack(side="left", fill="both", expand=True)
