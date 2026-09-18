from gui.controller.data_collection_controller import DataCollectionController
import ttkbootstrap as ttk

from gui.controller.camera_feed_controller import CameraFeedController
from gui.controller.dartboard_controller import DartboardController
from gui.controller.dartgame_controller import DartGameController
from gui.controller.menu_controller import MenuController
from gui.controller.player_controller import PlayerController
from gui.controller.scorecard_controller import ScorecardController
from gui.controller.scorepad_controller import ScorepadController
from gui.controller.source_controller import SourceController
from gui.events.event_channel import EventChannel
from gui.model.args import AppConfig
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class AppController(BaseController):
    """
    Coordinates the application state, user interface, and background processes.
    """

    def __init__(self, view: AppView, model: AppModel, config: AppConfig):
        super().__init__(view, model, EventChannel())

        self.config = config
        self.source_controller = SourceController(view, model, self._event_channel)
        self.dartboard_controller = DartboardController(
            view, model, self._event_channel
        )
        self.dartgame_controller = DartGameController(view, model, self._event_channel)
        self.player_controller = PlayerController(view, model, self._event_channel)
        self.camera_feed_controller = CameraFeedController(
            view, model, self._event_channel
        )
        self.scorepad_controller = ScorepadController(view, model, self._event_channel)
        self.menu_controller = MenuController(view, model, self._event_channel)
        self.scorecard_controller = ScorecardController(
            view, model, self._event_channel
        )
        self.data_collection_controller = DataCollectionController(
            view, model, self._event_channel
        )

        self.bind_menu()
        self.bind_components()

    def bind_menu(self) -> None:
        source_menu = self._create_menu("Source")
        self.source_controller.bind_menu(source_menu)

        view_menu = self._create_menu("View")
        self.dartboard_controller.bind_menu(view_menu)
        self.camera_feed_controller.bind_menu(view_menu)

        game_menu = self._create_menu("Game")
        self.dartgame_controller.bind_menu(game_menu)
        self.scorepad_controller.bind_menu(game_menu)
        self.menu_controller.bind_menu(game_menu)
        self.scorecard_controller.bind_menu(game_menu)
        self.player_controller.bind_menu(game_menu)

        data_menu = self._create_menu("Data")
        self.data_collection_controller.bind_menu(data_menu)

    def bind_components(self) -> None:
        self.source_controller.bind_components()
        self.dartboard_controller.bind_components()
        self.camera_feed_controller.bind_components()
        self.dartgame_controller.bind_components()
        self.player_controller.bind_components()
        self.scorepad_controller.bind_components()
        self.menu_controller.bind_components()
        self.scorecard_controller.bind_components()
        self.data_collection_controller.bind_components()

    def start(self):
        if self.config.source is not None:
            self.source_controller.set_source_by_value(self.config.source)

        self.source_controller.start()
        self.dartboard_controller.start()
        self.camera_feed_controller.start()
        self.dartgame_controller.start()
        self.player_controller.start()
        self.scorepad_controller.start()
        self.menu_controller.start()
        self.scorecard_controller.start()
        self.data_collection_controller.start()

        # start the main loop of the Tkinter application
        self._view.mainloop()

    def _create_menu(self, title: str) -> None:
        """Create the menu bar and attach it to the main window."""
        menu_btn = ttk.Menubutton(self._view.menu_frame, text=title)
        menu_btn.pack(side="left", padx=2)
        menu = ttk.Menu(menu_btn, tearoff=False)
        menu_btn["menu"] = menu
        return menu
