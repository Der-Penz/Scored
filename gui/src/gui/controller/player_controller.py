import tkinter as tk

from gui.events.event_channel import EventChannel
from gui.events.event_types import (
    GameStarted,
    PlayerAdded,
    PlayerRemoved,
    ScoreChanged,
    TurnChanged,
)
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class PlayerController(BaseController):
    """Thin controller for the PlayerView to keep scores displayed correctly"""

    def __init__(
        self, view: AppView, model: AppModel, event_channel: EventChannel
    ) -> None:
        super().__init__(view, model, event_channel)
        self.player_view = view.game_view.player_view

    def bind_menu(self, menu: tk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        self._event_channel.subscribe(PlayerAdded, self._on_player_added)
        self._event_channel.subscribe(PlayerRemoved, self._on_player_removed)
        self._event_channel.subscribe(ScoreChanged, self._on_score_changed)
        self._event_channel.subscribe(TurnChanged, self._on_turn_changed)
        self._event_channel.subscribe(GameStarted, self._on_game_started)

    def _on_player_added(self, event: PlayerAdded) -> None:
        self.player_view.add_player(event.player)

    def _on_player_removed(self, event: PlayerRemoved) -> None:
        self.player_view.remove_player(event.player)

    def _on_score_changed(self, _event: ScoreChanged) -> None:
        self._sync_scores()

    def _on_turn_changed(self, _event: TurnChanged) -> None:
        game = self._model.game
        if game is not None:
            self.player_view.set_current(game.current_player)

    def _on_game_started(self, _event: GameStarted) -> None:
        for player in self._model.players:
            if player not in self.player_view._player_widgets:
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
