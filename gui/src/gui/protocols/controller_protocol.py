from typing import Protocol
import tkinter as tk

from gui.model.model import AppModel
from gui.view.app_view import AppView


class ControllerProtocol(Protocol):
    def __init__(self, view: AppView, model: AppModel) -> None:
        """
        Initialize the controller with the given view and model.

        Parameters
        ----------
        view : AppView
            global view of the application
        model : AppModel
            global model of the application
        """
        ...

    def bind_menu(self, menu_bar: tk.Menu) -> None:
        """Bind menu events for the controller."""
        ...

    def bind_components(self) -> None:
        """Bind component events for the controller."""
        ...

    def start(self) -> None:
        """
        Start any background processes or threads required by the controller.
        This must be non-blocking and should return immediately. The controller should handle its own threading if necessary.
        """
        ...
