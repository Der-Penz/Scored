from abc import ABC, abstractmethod
from collections.abc import Callable

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import PanelRebound
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.view.app_view import AppView


class BaseController(ABC):
    def __init__(
        self, view: AppView, model: AppModel, event_channel: EventChannel
    ) -> None:
        """
        Initialize the controller with the given view and model.

        Parameters
        ----------
        view : AppView
            global view of the application
        model : AppModel
            global model of the application
        event_channel : EventChannel
            shared event bus for inter-controller communication
        """
        self._view = view
        self._model = model
        self._event_channel = event_channel

    def on_panel_rebound(self, panel: Panel, handler: Callable[[], None]) -> None:
        """
        Call *handler* whenever *panel* is rebuilt because it changed host.

        Use this to re-acquire the live widgets of *panel* from the view and to
        put them back into the state the model is in.
        """
        self._event_channel.subscribe(
            PanelRebound, lambda event: handler() if event.panel is panel else None
        )

    @abstractmethod
    def bind_menu(self, menu: ttk.Menu) -> None:
        """
        Bind menu commands to the given Menu

        Parameters
        ----------
        menu : ttk.Menu
            The parent Menu to which the menu commands will be bound.
        """
        ...

    @abstractmethod
    def bind_components(self) -> None:
        """Bind component events for the controller."""
        ...

    @abstractmethod
    def start(self) -> None:
        """
        Start any background processes or threads required by the controller.
        This must be non-blocking and should return immediately. The controller should handle its own threading if necessary.
        """
        ...
