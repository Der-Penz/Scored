import tkinter as tk

from gui.controller.controller import Controller
from gui.controller.camera_feed_controller import CameraFeedController
from gui.controller.dartboard_controller import DartboardController
from gui.model.model import AppModel
from gui.view.app_view import AppView
from gui.model.args import AppConfig


class AppController(Controller):
    """
    Coordinates the application state, user interface, and background processes.
    """

    def __init__(self, view: AppView, model: AppModel, config: AppConfig):
        self.model = model
        self.view = view
        self.config = config

        self.menu_bar = tk.Menu(self.view.master)
        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.source_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)
        self.menu_bar.add_cascade(label="Source", menu=self.source_menu)
        self.view.master.config(menu=self.menu_bar)

        self.dartboard_controller = DartboardController(view, model, self.view_menu)
        self.camera_feed_controller = CameraFeedController(
            view, model, self.view_menu, self.source_menu
        )

        self.bind_menu()
        self.bind_components()

    def bind_menu(self) -> None:
        self.dartboard_controller.bind_menu()
        self.camera_feed_controller.bind_menu()

    def bind_components(self) -> None:
        self.dartboard_controller.bind_components()
        self.camera_feed_controller.bind_components()

    def start(self):
        if self.config.source is not None:
            self.camera_feed_controller.set_source_by_value(self.config.source)

        self.dartboard_controller.start()
        self.camera_feed_controller.start()

        # start the main loop of the Tkinter application
        self.view.mainloop()
