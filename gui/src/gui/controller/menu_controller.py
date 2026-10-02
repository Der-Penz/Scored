import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import GameEnded, GameStarted, GameStateChanged, TurnChanged
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class MenuController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

    def bind_menu(self, _: ttk.Menu) -> None:
        self.status_label = ttk.Label(self._view.menu_frame, text="Scored")
        self.status_label.pack(side="right", padx=10, pady=2)

    def bind_components(self) -> None:
        self._event_channel.subscribe(GameStarted, lambda _: self.set_normal_status())
        self._event_channel.subscribe(
            GameStateChanged, lambda _: self.set_normal_status()
        )
        self._event_channel.subscribe(TurnChanged, lambda _: self.set_normal_status())
        self._event_channel.subscribe(GameEnded, lambda _: self.on_game_ended())

    def on_game_ended(self) -> None:
        assert self._model.game is not None, (
            "GameEnded event received but no game is active"
        )
        assert self._model.game.winner is not None, (
            "GameEnded event received but no winner is set"
        )
        self.set_status(
            f"{self._model.game.winner.name} finished the game! Waiting for new game to start..."
        )

    def start(self) -> None:
        pass

    def set_status(self, msg: str) -> None:
        self.status_label.config(text=msg)

    def set_normal_status(self) -> None:
        assert self._model.game is not None, (
            "GameEnded event received but no game is active"
        )

        round = f"Round: {self._model.game.current_leg.round} | Throw: {self._model.game.current_leg.throw}"
        rule = f"Opening: {self._model.game.start_rule.value} | Finish: {self._model.game.finish_rule.value} | Starting: {self._model.game.starting_score}"
        msg = f"{rule} | {round}"
        self.status_label.config(text=msg)
