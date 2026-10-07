import tkinter as tk

from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier

from gui.events.event_channel import EventChannel
from gui.events.event_types import (
    DartThrowEvent,
    UndoRequested,
)
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from gui.view.scorepad_view import ScorepadView


class ScorepadController(BaseController):
    """Thin controller for the ScorepadView to keep multiplier state and expose on_throw."""

    def __init__(
        self, view: AppView, model: AppModel, event_channel: EventChannel
    ) -> None:
        super().__init__(view, model, event_channel)
        self.multiplier = Multiplier.SINGLE

    @property
    def scorepad_view(self) -> ScorepadView:
        """The live scorepad; the control panel is rebuilt when it changes host."""
        return self._view.game_view.scorepad_view

    def bind_menu(self, menu: tk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        self._view.master.bind("<BackSpace>", lambda _: self._undo_throw())

        self._view.master.bind(
            "<Shift_L>", lambda _: self.set_multiplier(Multiplier.DOUBLE)
        )
        self._view.master.bind(
            "<Control_L>", lambda _: self.set_multiplier(Multiplier.TRIPLE)
        )

        self.on_panel_rebound(Panel.CONTROL, self._restore_scorepad)

    def _bind_buttons(self) -> None:
        """Point the scorepad buttons at this controller."""
        scorepad = self.scorepad_view

        for num, btn in scorepad.number_buttons.items():
            btn.config(command=lambda n=num: self.on_number_press(n))

        scorepad.double_btn.config(
            command=lambda: self.set_multiplier(Multiplier.DOUBLE)
        )
        scorepad.triple_btn.config(
            command=lambda: self.set_multiplier(Multiplier.TRIPLE)
        )
        scorepad.bull25.config(
            command=lambda: self.on_special_press(Multiplier.OUTER_BULL)
        )
        scorepad.bull50.config(
            command=lambda: self.on_special_press(Multiplier.INNER_BULL)
        )
        scorepad.miss.config(command=lambda: self.on_special_press(Multiplier.MISS))

    def _restore_scorepad(self) -> None:
        """Re-hook and repaint the scorepad after the control panel was rebuilt."""
        self._bind_buttons()
        self.scorepad_view.highlight_multiplier(self.multiplier)

    def _undo_throw(self) -> None:
        self._event_channel.emit(UndoRequested())

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
        self._restore_scorepad()
