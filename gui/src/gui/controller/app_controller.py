import logging
import tkinter as tk

import ttkbootstrap as ttk

from gui.controller.annotation_controller import AnnotationController
from gui.controller.camera_feed_controller import CameraFeedController
from gui.controller.dartboard_controller import DartboardController
from gui.controller.dartgame_controller import DartGameController
from gui.controller.data_collection_controller import DataCollectionController
from gui.controller.image_collection_controller import ImageCollectionController
from gui.controller.label_controller import LabelController
from gui.controller.layout_controller import LayoutController
from gui.controller.logging_controller import LoggingController
from gui.controller.menu_controller import MenuController
from gui.controller.player_controller import PlayerController
from gui.controller.scorecard_controller import ScorecardController
from gui.controller.scorepad_controller import ScorepadController
from gui.controller.source_controller import SourceController
from gui.events.event_types import ModeChanged
from gui.model.args import AppConfig, AppMode
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView

# Menus that only make sense for one of the two modes.
GAME_ONLY_MENUS = ("Source", "Game", "Data")
LABEL_ONLY_MENUS = ("Labels",)


class AppController(BaseController):
    """
    Coordinates the application state, user interface, and background processes.
    """

    def __init__(self, view: AppView, model: AppModel, config: AppConfig):
        # The view owns the channel, because the panel manager announces panel
        # rebuilds on it while the controllers are still being constructed.
        super().__init__(view, model, view.event_channel)

        self.config = config
        self._menu_buttons: dict[str, ttk.Menubutton] = {}
        self._menu_order: list[str] = []
        self._mode_var = tk.StringVar(value=model.mode.value)

        self.source_controller = SourceController(view, model, self._event_channel)
        self.layout_controller = LayoutController(view, model, self._event_channel)
        self.dartboard_controller = DartboardController(view, model, self._event_channel)
        self.dartgame_controller = DartGameController(view, model, self._event_channel)
        self.player_controller = PlayerController(view, model, self._event_channel)
        self.camera_feed_controller = CameraFeedController(view, model, self._event_channel)
        self.scorepad_controller = ScorepadController(view, model, self._event_channel)
        self.menu_controller = MenuController(view, model, self._event_channel)
        self.scorecard_controller = ScorecardController(view, model, self._event_channel)
        self.image_collection_controller = ImageCollectionController(
            view, model, self._event_channel
        )
        self.annotation_controller = AnnotationController(view, model, self._event_channel)
        self.label_controller = LabelController(view, model, self._event_channel)
        self.data_collection_controller = DataCollectionController(view, model, self._event_channel)
        self.logging_controller = LoggingController(view, model, self._event_channel)

        self.layout_controller.add_visibility_listener(
            Panel.SOURCE, self.camera_feed_controller.set_active
        )

        self.bind_menu()
        self.bind_components()
        self._apply_mode(model.mode)

    def bind_menu(self) -> None:
        mode_menu = self._create_menu("Mode")
        for mode in AppMode:
            mode_menu.add_radiobutton(
                label=mode.label,
                value=mode.value,
                variable=self._mode_var,
                command=lambda chosen=mode: self.set_mode(chosen),
            )

        source_menu = self._create_menu("Source")
        self.source_controller.bind_menu(source_menu)

        view_menu = self._create_menu("View")
        self.layout_controller.bind_menu(view_menu)

        game_menu = self._create_menu("Game")
        self.dartgame_controller.bind_menu(game_menu)
        self.scorepad_controller.bind_menu(game_menu)
        self.menu_controller.bind_menu(game_menu)
        self.scorecard_controller.bind_menu(game_menu)
        self.player_controller.bind_menu(game_menu)

        label_menu = self._create_menu("Labels")
        self.label_controller.bind_menu(label_menu)

        data_menu = self._create_menu("Data")
        self.data_collection_controller.bind_menu(data_menu)

        log_menu = self._create_menu("Logs")
        self.logging_controller.bind_menu(log_menu)

    def bind_components(self) -> None:
        self.layout_controller.bind_components()
        self.source_controller.bind_components()
        self.dartboard_controller.bind_components()
        self.camera_feed_controller.bind_components()
        self.dartgame_controller.bind_components()
        self.player_controller.bind_components()
        self.scorepad_controller.bind_components()
        self.menu_controller.bind_components()
        self.scorecard_controller.bind_components()
        self.image_collection_controller.bind_components()
        self.annotation_controller.bind_components()
        self.label_controller.bind_components()
        self.data_collection_controller.bind_components()
        self.logging_controller.bind_components()

    def start(self):
        if self.config.source is not None and self._model.mode is AppMode.GAME:
            self.source_controller.set_source_by_value(self.config.source)

        self.source_controller.start()
        self.dartboard_controller.start()
        self.camera_feed_controller.start()
        self.dartgame_controller.start()
        self.player_controller.start()
        self.scorepad_controller.start()
        self.layout_controller.start()
        self.menu_controller.start()
        self.scorecard_controller.start()
        self.image_collection_controller.start()
        self.annotation_controller.start()
        self.label_controller.start()
        self.data_collection_controller.start()
        self.logging_controller.start()

        # start the main loop of the Tkinter application
        self._view.mainloop()

        # persist any labels that are still held in memory
        self.annotation_controller.save_round()
        self.label_controller.save_if_dirty()

    def set_mode(self, mode: AppMode) -> None:
        """Switch the application between playing a game and labeling images.

        The controllers that do not belong to *mode* stay constructed but idle,
        so switching back keeps the game state, the open folder and the labels
        that are still unsaved in place.
        """
        if mode is self._model.mode:
            return
        logging.info(f"Switching to {mode.label} mode")
        self._model.mode = mode
        self._mode_var.set(mode.value)
        self._event_channel.emit(ModeChanged(mode=mode))
        self._apply_mode(mode)

    def _apply_mode(self, mode: AppMode) -> None:
        """Show the menus that belong to *mode* and hide the others."""
        hidden = GAME_ONLY_MENUS if mode is AppMode.LABEL else LABEL_ONLY_MENUS
        for title in self._menu_order:
            button = self._menu_buttons[title]
            button.pack_forget()
            if title not in hidden:
                button.pack(side="left", padx=2)
        self._view.master.title(f"Scored GUI — {mode.label}")

    def _create_menu(self, title: str) -> ttk.Menu:
        """Create the menu bar and attach it to the main window."""
        menu_btn = ttk.Menubutton(self._view.menu_frame, text=title)
        menu_btn.pack(side="left", padx=2)
        menu = ttk.Menu(menu_btn, tearoff=False)
        menu_btn["menu"] = menu
        self._menu_buttons[title] = menu_btn
        self._menu_order.append(title)
        return menu
