import tkinter as tk

from gui.protocols.controller_protocol import ControllerProtocol
from gui.model.model import AppModel
from gui.view.app_view import AppView


class DartboardController(ControllerProtocol):
    def __init__(self, view: AppView, model: AppModel):
        super().__init__(view, model)
        self._view = view
        self._model = model

    def bind_menu(self, menu_bar: tk.Menu) -> None:
        self._dartboard_visible = tk.BooleanVar(value=True)
        menu_bar.add_checkbutton(
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
