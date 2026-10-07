import tkinter as tk

from gui.events.event_channel import EventChannel
from gui.events.event_types import (
    GameStarted,
    GameStateChanged,
    PlayerAdded,
    PlayerRemoved,
    TurnChanged,
)
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from gui.view.game.player_view import PlayerView


class PlayerController(BaseController):
    """Thin controller for the PlayerView to keep scores displayed correctly"""

    def __init__(
        self, view: AppView, model: AppModel, event_channel: EventChannel
    ) -> None:
        super().__init__(view, model, event_channel)

    @property
    def player_view(self) -> PlayerView:
        """The live player view; the control panel is rebuilt when it changes host."""
        return self._view.game_view.player_view

    def bind_menu(self, menu: tk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        self._event_channel.subscribe(PlayerAdded, self._on_player_added)
        self._event_channel.subscribe(PlayerRemoved, self._on_player_removed)
        self._event_channel.subscribe(GameStateChanged, self._on_game_state_changed)
        self._event_channel.subscribe(TurnChanged, self._on_turn_changed)
        self._event_channel.subscribe(GameStarted, self._on_game_started)
        self.on_panel_rebound(Panel.CONTROL, self._restore_players)

    def _on_player_added(self, event: PlayerAdded) -> None:
        self.player_view.add_player(event.player)

    def _on_player_removed(self, event: PlayerRemoved) -> None:
        self.player_view.remove_player(event.player)

    def _on_game_state_changed(self, _event: GameStateChanged) -> None:
        self._sync_scores()

    def _on_turn_changed(self, _event: TurnChanged) -> None:
        game = self._model.game
        if game is not None:
            self.player_view.set_current(game.current_player)

    def _on_game_started(self, _event: GameStarted) -> None:
        self._restore_players()

    def _restore_players(self) -> None:
        """Rebuild the player cards from the model and refresh the scores."""
        for player in self._model.players:
            self.player_view.add_player(player)
        self._sync_scores()

    def _sync_scores(self) -> None:
        game = self._model.game
        if game is None:
            return
        for player in self._model.players:
            leg = game.leg_for(player)
            self.player_view.update_player(player, score=leg.score, avg=leg.avg())
        self.player_view.set_current(game.current_player)

    def start(self) -> None:
        pass
