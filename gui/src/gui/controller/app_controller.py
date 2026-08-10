import tkinter as tk
import ttkbootstrap as ttk

from gui.controller.scorepad_controller import ScorepadController
from gui.protocols.controller_protocol import ControllerProtocol
from gui.controller.camera_feed_controller import CameraFeedController
from gui.controller.dartboard_controller import DartboardController
from gui.controller.dartgame_controller import DartGameController
from gui.controller.source_controller import SourceController
from gui.model.model import AppModel
from gui.view.app_view import AppView
from gui.model.args import AppConfig


class AppController(ControllerProtocol):
    """
    Coordinates the application state, user interface, and background processes.
    """

    def __init__(self, view: AppView, model: AppModel, config: AppConfig):
        self.model = model
        self.view = view
        self.config = config

        self.source_controller = SourceController(view, model)
        self.dartboard_controller = DartboardController(view, model)
        self.dartgame_controller = DartGameController(view, model)
        self.camera_feed_controller = CameraFeedController(
            view, model, self.source_controller
        )
        self.scorepad_controller = ScorepadController(view, model)

        menu_bar = ttk.Menu(self.view.master)
        self.bind_menu(menu_bar)
        self.bind_components()

    def bind_menu(self, menu_bar: tk.Menu) -> None:
        self.view.master.config(menu=menu_bar)

        self.source_controller.bind_menu(menu_bar)

        view_menu = ttk.Menu(menu_bar)
        menu_bar.add_cascade(label="View", menu=view_menu)

        self.dartboard_controller.bind_menu(view_menu)
        self.camera_feed_controller.bind_menu(view_menu)
        self.dartgame_controller.bind_menu(menu_bar)
        self.scorepad_controller.bind_menu(menu_bar)

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
