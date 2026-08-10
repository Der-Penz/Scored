import tkinter as tk

from gui.protocols.controller_protocol import ControllerProtocol
from gui.model.model import AppModel
from gui.view.app_view import AppView


class DartboardController(ControllerProtocol):
    def __init__(self, view: AppView, model: AppModel):
        super().__init__(view, model)
        self._view = view
        self._model = model
        self._left_view = view.left_view

    def bind_menu(self, menu: tk.Menu) -> None:
        self._dartboard_visible = tk.BooleanVar(value=True)
        menu.add_checkbutton(
            label="Show Dartboard",
            variable=self._dartboard_visible,
            command=self._toggle_dartboard_visibility,
        )

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        pass

    def _toggle_dartboard_visibility(self) -> None:
        self._left_view.set_dartboard_visible(self._dartboard_visible.get())
        self._view.refresh_left_panes()
