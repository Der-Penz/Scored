from gui.controller.controller import Controller
from gui.controller.source_manager import SourceManager
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
        self.source_manager = SourceManager(view, model)

        self.bind_menu()
        self.bind_components()

    def bind_menu(self) -> None:
        self.source_manager.bind_menu()

    def start(self):
        if self.config.source is not None:
            self.source_manager.set_source_by_value(self.config.source)

        self.source_manager.start()

        # start the main loop of the Tkinter application
        self.view.mainloop()
