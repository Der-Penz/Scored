import tkinter as tk

from gui.events.event_channel import EventChannel
from gui.events.event_types import DartThrowEvent, PlayerAddedEvent, PlayersChangedEvent
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier


class PlayerController(BaseController):
    """Thin controller for the PlayerView to keep scores displayed correctly"""

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel) -> None:
        super().__init__(view, model, event_channel)
        self.player_view = view.game_view.player_view

    def bind_menu(self, menu: tk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        self._event_channel.subscribe(PlayersChangedEvent, self._on_players_changed)
        self._event_channel.subscribe(PlayerAddedEvent, self._on_player_added)

    def _on_players_changed(self, _event: PlayersChangedEvent) -> None:
        self.player_view.set_current(self._model.players)
 

    def start(self) -> None:
        pass
