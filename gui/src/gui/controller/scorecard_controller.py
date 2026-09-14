import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from gui.view.scorecard_view import ScorecardView


class ScorecardController(BaseController):
    """
    Opens a non-reactive snapshot scorecard of the current game's throw history.

    The window is built once from the model at open time and is not subscribed
    to any events; rebuild it (menu item or Ctrl+T) to see fresh data.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._scorecard_window: ScorecardView | None = None

    def bind_menu(self, menu: ttk.Menu) -> None:
        menu.add_command(
            label="Scorecard",
            command=self._open_scorecard,
            accelerator="Ctrl+T",
        )

    def bind_components(self) -> None:
        self._view.master.bind_all("<Control-t>", lambda _: self._open_scorecard())

    def start(self) -> None:
        pass

    def _open_scorecard(self) -> None:
        if self._scorecard_window is not None and self._scorecard_window.winfo_exists():
            self._scorecard_window.destroy()

        self._scorecard_window = ScorecardView(
            master=self._view,
            players=self._model.players,
            game=self._model.game,
        )
