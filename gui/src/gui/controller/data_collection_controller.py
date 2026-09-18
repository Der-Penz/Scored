from pathlib import Path
import platform
import subprocess
import tkinter as tk
from tkinter import filedialog

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class DataCollectionController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        self._output_dir: Path | None = (
            Path(model.args.data_dir) if model.args.data_dir else None
        )

        self._enabled_var = tk.BooleanVar(value=False)
        self._skip_busts_var = tk.BooleanVar(value=False)
        self._replace_on_edit_var = tk.BooleanVar(value=False)

    def bind_menu(self, menu: ttk.Menu) -> None:
        """Attach the data collection commands to the ``Data`` menu."""
        menu.add_checkbutton(
            label="Enable Data Collection",
            variable=self._enabled_var,
            command=self._on_enabled_toggle,
        )

        menu.add_separator()
        menu.add_checkbutton(
            label="Skip capturing bust throws",
            variable=self._skip_busts_var,
        )
        menu.add_checkbutton(
            label="Replace image when dart is edited",
            variable=self._replace_on_edit_var,
        )
        menu.add_separator()
        menu.add_command(label="Open Output Folder", command=self._open_output_folder)

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        """Nothing to start; all work is driven by events."""

    def _on_enabled_toggle(self) -> None:
        if not self._enabled_var.get():
            return

        if not self._output_dir:
            directory = filedialog.askdirectory(
                title="Choose data collection directory", parent=self._view
            )
            if directory:
                self._output_dir = Path(directory)
            else:
                self._enabled_var.set(False)
                return

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as _:
            ttk.Messagebox.show_error(
                message=f"Could not create the data collection directory.",
                title="Data Collection Error",
                parent=self._view,
            )
            self._enabled_var.set(False)
            return

    def _open_output_folder(self) -> None:
        """Open the output directory in the system file manager."""
        if not self._output_dir:
            ttk.Messagebox.show_info(
                message="Choose an output directory first.",
                title="Data Collection",
                parent=self._view,
            )
            return
        if platform.system() == "Windows":
            program = "explorer"
        elif platform.system() == "Darwin":
            program = "open"
        else:
            program = "xdg-open"
        subprocess.run([program, str(self._output_dir)])
