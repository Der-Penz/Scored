import tkinter as tk

from gui.events.event_channel import EventChannel
from gui.events.event_types import ScoreChanged, TurnChanged
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class DartboardController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._left_view = view.left_view

    def bind_menu(self, menu: tk.Menu) -> None:
        self._dartboard_visible = tk.BooleanVar(value=True)
        menu.add_checkbutton(
            label="Show Dartboard",
            variable=self._dartboard_visible,
            command=self._toggle_dartboard_visibility,
        )

    def bind_components(self) -> None:
        self._event_channel.subscribe(TurnChanged, lambda _: self.clear())
        self._event_channel.subscribe(ScoreChanged, lambda _: self.on_score_changed)

    def clear(self) -> None:
        #TODO clear out all drawn darts
        pass
        
    def on_score_changed(self, _) -> None:
        leg = self._model.game.current_leg
        
        for throw in leg.current_turn_throws:
            #TODO draw the dart throw on the dartboard and make them draggable. on drag release update the throw in the model and redraw the dartboard
            pass
            

    def start(self) -> None:
        pass

    def _toggle_dartboard_visibility(self) -> None:
        self._left_view.set_dartboard_visible(self._dartboard_visible.get())
        self._view.refresh_left_panes()
