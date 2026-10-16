from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import DataCollectionChanged
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class DataCollectionController(BaseController):
    """Owns the Data menu and toggles whether annotated data is collected.

    Data collection is enabled whenever a directory to store the data in is
    set: ``--data_dir`` enables it at startup, and the ``Data`` menu either
    selects a directory at runtime or removes it again to stop collecting.
    The controllers that collect the frames and write the annotations follow
    the :class:`~gui.model.model.AppModel` and the
    :class:`~gui.events.event_types.DataCollectionChanged` event, so disabling
    here stops the collection itself and not just the disk writes.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._menu: ttk.Menu | None = None
        self._status_index: int | None = None
        self._remove_index: int | None = None

    def bind_menu(self, menu: ttk.Menu) -> None:
        self._menu = menu
        menu.add_command(label="Data Collection: Off")
        self._status_index = menu.index("end")
        menu.add_separator()
        menu.add_command(label="Select Data Directory…", command=self._on_select_directory)
        menu.add_command(
            label="Remove Data Directory",
            command=self._on_remove_directory,
        )
        self._remove_index = menu.index("end")
        self._update_menu()

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        """Enable collection for a directory that was passed on the CLI."""
        directory = self._model.data_dir
        if directory is not None:
            self._enable(directory)

    def _on_select_directory(self) -> None:
        directory = filedialog.askdirectory(
            title="Select data collection directory", parent=self._view
        )
        if not directory:
            return
        self._enable(Path(directory))

    def _on_remove_directory(self) -> None:
        if not self._model.data_collection_enabled:
            return
        self._disable()

    def _enable(self, directory: Path) -> None:
        """Store collected data in *directory*, creating it if needed."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logging.error(f"Could not create the data collection directory: {error}")
            ttk.Messagebox.show_error(
                message="Could not create the data collection directory.",
                title="Data Collection Error",
                parent=self._view,
            )
            self._disable()
            return

        self._model.data_dir = directory
        logging.info(f"Collecting data in {directory}")
        self._update_menu()
        self._event_channel.emit(DataCollectionChanged(True, directory))

    def _disable(self) -> None:
        """Stop collecting data; nothing is written until a directory is set."""
        self._model.data_dir = None
        logging.info("Data collection disabled")
        self._update_menu()
        self._event_channel.emit(DataCollectionChanged(False, None))

    def _update_menu(self) -> None:
        """Show the current state and gate the remove entry in the Data menu."""
        if self._menu is None or self._status_index is None or self._remove_index is None:
            return
        directory = self._model.data_dir
        status = f"Data Collection: {directory}" if directory else "Data Collection: Off"
        self._menu.entryconfigure(self._status_index, label=status)
        self._menu.entryconfigure(self._remove_index, state="normal" if directory else "disabled")
