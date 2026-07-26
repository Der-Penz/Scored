import tkinter as tk

from gui.controller.controller import Controller
from gui.model.model import AppModel
from gui.view.app_view import AppView


class DartboardController(Controller):
    def __init__(self, view: AppView, model: AppModel, view_menu):
        super().__init__(view, model)
        self._view = view
        self._model = model
        self._view_menu = view_menu

    def bind_menu(self) -> None:
        self._dartboard_visible = tk.BooleanVar(value=True)
        self._view_menu.add_checkbutton(
            label="Show Dartboard",
            variable=self._dartboard_visible,
            command=self._toggle_dartboard_visibility,
        )

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        pass

    def _toggle_dartboard_visibility(self) -> None:
        self._view.set_dartboard_visible(self._dartboard_visible.get())

    def addDartThrow(self, _dart_throw) -> None:
        pass

    def resetDartThrow(self) -> None:
        pass