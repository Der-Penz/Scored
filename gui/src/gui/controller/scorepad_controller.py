import tkinter as tk

from gui.events.event_channel import EventChannel
from gui.events.event_types import DartThrowEvent
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier


class ScorepadController(BaseController):
    """Thin controller for the ScorepadView to keep multiplier state and expose on_throw."""

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel) -> None:
        super().__init__(view, model, event_channel)
        self.scorepad_view = view.game_view.scorepad_view

        self.multiplier = Multiplier.SINGLE

    def bind_menu(self, menu: tk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        for num, btn in self.scorepad_view.number_buttons.items():
            btn.config(command=lambda n=num: self.on_number_press(n))

        self.scorepad_view.double_btn.config(
            command=lambda: self.set_multiplier(Multiplier.DOUBLE)
        )
        self.scorepad_view.triple_btn.config(
            command=lambda: self.set_multiplier(Multiplier.TRIPLE)
        )
        self.scorepad_view.bull25.config(
            command=lambda: self.on_special_press(Multiplier.OUTER_BULL)
        )
        self.scorepad_view.bull50.config(
            command=lambda: self.on_special_press(Multiplier.INNER_BULL)
        )
        self.scorepad_view.miss.config(
            command=lambda: self.on_special_press(Multiplier.MISS)
        )

    def set_multiplier(self, multiplier: Multiplier) -> None:
        if multiplier not in (Multiplier.SINGLE, Multiplier.DOUBLE, Multiplier.TRIPLE):
            # invalid multiplier, ignore
            self.multiplier = Multiplier.SINGLE
        if multiplier == self.multiplier:
            # toggle off if already selected
            self.multiplier = Multiplier.SINGLE
        else:
            self.multiplier = multiplier
        self.scorepad_view.highlight_multiplier(self.multiplier)

    def on_special_press(self, multiplier: Multiplier) -> None:
        if multiplier in (Multiplier.SINGLE, Multiplier.DOUBLE, Multiplier.TRIPLE):
            return
        dart_throw = DartThrow(number=0, multiplier=multiplier)
        self.set_multiplier(Multiplier.SINGLE)  # reset multiplier after throw

        self._event_channel.emit(DartThrowEvent(dart_throw))

    def on_number_press(self, number: int) -> None:
        dart_throw = DartThrow(number=number, multiplier=self.multiplier)
        self.set_multiplier(Multiplier.SINGLE)
        self._event_channel.emit(DartThrowEvent(dart_throw))

    def start(self) -> None:
        pass
