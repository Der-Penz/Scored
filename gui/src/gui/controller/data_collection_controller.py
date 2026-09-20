from pathlib import Path
import platform
import subprocess
import tkinter as tk
from tkinter import filedialog

from gui.events.event_types import FrameCapturedEvent, ScoreChanged
import numpy as np
import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel, Subscription
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class DataCollectionController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        self._output_dir: Path | None = (
            Path(model.args.data_dir) if model.args.data_dir else None
        )
        self.events: list[Subscription] = []
        self.current_frame: np.ndarray | None = None

        self._enabled_var = tk.BooleanVar(value=self._output_dir is not None)
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
        if self._enabled_var.get():
            self._enable_data_collection()

    def start(self) -> None:
        """Nothing to start; all work is driven by events."""

    def _on_frame_captured(self, event: FrameCapturedEvent) -> None:
        self.current_frame = event.frame

    def _on_enabled_toggle(self) -> None:
        if not self._enabled_var.get():
            self._enabled_var.set(False)
            self._disable_data_collection()
            return

        if not self._output_dir:
            directory = filedialog.askdirectory(
                title="Choose data collection directory", parent=self._view
            )
            if directory:
                self._output_dir = Path(directory)
            else:
                self._enabled_var.set(False)
                self._disable_data_collection()
                return

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._enable_data_collection()
        except OSError as _:
            ttk.Messagebox.show_error(
                message=f"Could not create the data collection directory.",
                title="Data Collection Error",
                parent=self._view,
            )
            self._enabled_var.set(False)
            self._disable_data_collection()
            return

    def _on_score_changed(self) -> None:
        print("Saving frame to output directory")

    def _disable_data_collection(self) -> None:
        for subscription in self.events:
            subscription.unsubscribe()
        self.events = []

    def _enable_data_collection(self) -> None:
        self._disable_data_collection()  # remove any existing subscriptions
        self.events.append(
            self._event_channel.subscribe(
                FrameCapturedEvent, lambda e: self._on_frame_captured(e)
            )
        )
        self.events.append(
            self._event_channel.subscribe(
                ScoreChanged, lambda _: self._on_score_changed()
            )
        )

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
