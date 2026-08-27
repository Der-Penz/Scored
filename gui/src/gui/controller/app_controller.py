import ttkbootstrap as ttk

from gui.controller.camera_feed_controller import CameraFeedController
from gui.controller.dartboard_controller import DartboardController
from gui.controller.dartgame_controller import DartGameController
from gui.controller.scorepad_controller import ScorepadController
from gui.controller.source_controller import SourceController
from gui.events.event_channel import EventChannel
from gui.model.args import AppConfig
from gui.model.model import AppModel
from gui.view.app_view import AppView


class AppController():
    """
    Coordinates the application state, user interface, and background processes.
    """

    def __init__(self, view: AppView, model: AppModel, config: AppConfig):
        self.event_channel = EventChannel()

        self.source_controller = SourceController(view, model, self.event_channel)
        self.dartboard_controller = DartboardController(view, model, self.event_channel)
        self.dartgame_controller = DartGameController(view, model, self.event_channel)
        self.camera_feed_controller = CameraFeedController(view, model, self.event_channel)
        self.scorepad_controller = ScorepadController(view, model, self.event_channel)

        self.bind_menu()
        self.bind_components()

    def bind_menu(self) -> None:
        source_btn = ttk.Menubutton(self.view.menu_frame, text="Source")
        source_btn.pack(side="left", padx=2)
        source_menu = ttk.Menu(source_btn, tearoff=False)
        source_btn["menu"] = source_menu

        self.source_controller.bind_menu(source_menu)

        view_btn = ttk.Menubutton(self.view.menu_frame, text="View")
        view_btn.pack(side="left", padx=2)
        view_menu = ttk.Menu(view_btn, tearoff=False)
        view_btn["menu"] = view_menu

        self.dartboard_controller.bind_menu(view_menu)
        self.camera_feed_controller.bind_menu(view_menu)

        game_btn = ttk.Menubutton(self.view.menu_frame, text="Game")
        game_btn.pack(side="left", padx=2)
        game_menu = ttk.Menu(game_btn, tearoff=False)
        game_btn["menu"] = game_menu

        self.dartgame_controller.bind_menu(game_menu)

        self.scorepad_controller.bind_menu(game_menu)

    def bind_components(self) -> None:
        self.source_controller.bind_components()
        self.dartboard_controller.bind_components()
        self.camera_feed_controller.bind_components()
        self.dartgame_controller.bind_components()
        self.scorepad_controller.bind_components()

    def start(self):
        if self.config.source is not None:
            self.source_controller.set_source_by_value(self.config.source)

        self.source_controller.start()
        self.dartboard_controller.start()
        self.camera_feed_controller.start()
        self.dartgame_controller.start()
        self.scorepad_controller.start()

        # start the main loop of the Tkinter application
        self.view.mainloop()
