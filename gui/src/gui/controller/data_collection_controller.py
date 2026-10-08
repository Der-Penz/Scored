import logging
import tkinter as tk
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import numpy as np
import ttkbootstrap as ttk
from scored_lib.annotation.image_annotation import ImageAnnotation
from scored_lib.annotation.leg_annotation import LegAnnotation
from scored_lib.annotation.leg_annotation_handler import LegAnnotationHandler

from gui.events.event_channel import EventChannel, Subscription
from gui.events.event_types import (
    FrameCapturedEvent,
    GameStarted,
    ThrowEdited,
    ThrowRecorded,
    ThrowRemoved,
)
from gui.helper import open_in_file_manager
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class DataCollectionController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        self._output_dir: Path | None = (
            Path(model.config.data_dir) if model.config.data_dir else None
        )

        if self._output_dir:
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Collecting data in {self._output_dir}")
            except OSError as error:
                logging.error(
                    f"Could not create the data collection directory: {error}"
                )
                ttk.Messagebox.show_error(
                    message="Could not create the data collection directory.",
                    title="Data Collection Error",
                    parent=self._view,
                )
                self._output_dir = None

        self.events: list[Subscription] = []
        self.current_frame: np.ndarray | None = None
        self.annotation_handler: dict[str, LegAnnotationHandler] | None = None

        self._enabled_var = tk.BooleanVar(value=self._output_dir is not None)
        self._replace_on_edit_var = tk.BooleanVar(value=False)
        self._delete_removed_var = tk.BooleanVar(value=True)

    def bind_menu(self, menu: ttk.Menu) -> None:
        """Attach the data collection commands to the ``Data`` menu."""
        menu.add_checkbutton(
            label="Enable Data Collection",
            variable=self._enabled_var,
            command=self._on_enabled_toggle,
        )

        menu.add_separator()
        menu.add_checkbutton(
            label="Replace image when dart is edited",
            variable=self._replace_on_edit_var,
        )
        menu.add_checkbutton(
            label="Delete removed throws",
            variable=self._delete_removed_var,
        )
        menu.add_separator()
        menu.add_command(label="Open Output Folder", command=self._open_output_folder)

    @property
    def enabled(self) -> bool:
        return self._enabled_var.get()

    def bind_components(self) -> None:
        if self.enabled:
            self._enable_data_collection()

    def start(self) -> None:
        if self.enabled:
            self._enable_data_collection()

    def _on_frame_captured(self, event: FrameCapturedEvent) -> None:
        self.current_frame = event.frame

    def _on_enabled_toggle(self) -> None:
        if not self.enabled:
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
            logging.info(f"Data collection enabled, writing to {self._output_dir}")
        except OSError as error:
            logging.error(f"Could not create the data collection directory: {error}")
            ttk.Messagebox.show_error(
                message="Could not create the data collection directory.",
                title="Data Collection Error",
                parent=self._view,
            )
            self._enabled_var.set(False)
            self._disable_data_collection()
            return

    def _handler_for(self, player_id: str) -> LegAnnotationHandler | None:
        """Get the annotation handler of a player, if collection is running."""
        if self.annotation_handler is None:
            return None
        return self.annotation_handler.get(player_id)

    def _annotation_for(
        self, event: ThrowRecorded | ThrowEdited, leg_id: str
    ) -> ImageAnnotation:
        """Build the on-disk annotation for a recorded or edited throw."""
        result = event.result
        return ImageAnnotation(
            is_bust=result.bust,
            leg_id=leg_id,
            round=result.round,
            throw=result.throw,
        )

    def _on_throw_recorded(self, event: ThrowRecorded) -> None:
        handler = self._handler_for(event.player.id)
        if handler is None:
            return

        if self.current_frame is None:
            logging.warning(
                f"Throw recorded for {event.player.name} but no frame is available"
            )
            return

        handler.add(
            self._annotation_for(event, handler.info.leg_id), self.current_frame
        )

    def _on_throw_edited(self, event: ThrowEdited) -> None:
        handler = self._handler_for(event.player.id)
        if handler is None:
            return

        result = event.result
        if not handler.exists(result.round, result.throw):
            # Nothing stored yet, e.g. no camera frame arrived for this throw.
            logging.warning(
                f"Throw edited for {event.player.name} but no sample was stored for {result.round}_{result.throw} yet"
            )
            return

        image = self.current_frame if self._replace_on_edit_var.get() else None
        handler.edit(
            result.round,
            result.throw,
            annotation=self._annotation_for(event, handler.info.leg_id),
            image=image,
        )

    def _on_throw_removed(self, event: ThrowRemoved) -> None:
        handler = self._handler_for(event.player.id)
        if handler is None:
            return

        if self._delete_removed_var.get():
            handler.remove(event.round, event.throw)
            return

        if handler.mark_removed(event.round, event.throw) is not None:
            logging.info(
                f"Kept removed sample {event.round}_{event.throw} as a tombstone"
            )

    def _on_game_started(self) -> None:
        assert self._model.game is not None, (
            "GameStarted event received but model.game is None"
        )
        assert self._output_dir is not None, (
            "GameStarted event received but output_dir is None"
        )

        logging.info("Game started, initializing data collection.")

        self.annotation_handler = {}
        current = datetime.now()
        directory = self._output_dir / f"game_{current.strftime('%d_%m_%H_%M_%S')}"

        for player in self._model.players:
            leg_anno = LegAnnotation(
                leg_id=uuid.uuid4().hex,
                player_name=player.name,
                starting_score=self._model.game.starting_score,
            )

            annotation_handler = LegAnnotationHandler(
                directory=directory / player.name,
                info=leg_anno,
            )
            self.annotation_handler[player.id] = annotation_handler

        count = len(self._model.players)
        logging.info(f"Started annotations for {count} player(s) in {directory}")

    def _disable_data_collection(self) -> None:
        for subscription in self.events:
            subscription.cancel()
        self.events = []
        logging.info("Data collection disabled")

    def _enable_data_collection(self) -> None:
        self._disable_data_collection()  # remove any existing subscriptions
        self.events.append(
            self._event_channel.subscribe(
                FrameCapturedEvent, lambda e: self._on_frame_captured(e)
            )
        )
        self.events.append(
            self._event_channel.subscribe(
                ThrowRecorded, lambda evt: self._on_throw_recorded(evt)
            )
        )
        self.events.append(
            self._event_channel.subscribe(
                ThrowEdited, lambda evt: self._on_throw_edited(evt)
            )
        )
        self.events.append(
            self._event_channel.subscribe(
                ThrowRemoved, lambda evt: self._on_throw_removed(evt)
            )
        )
        self.events.append(
            self._event_channel.subscribe(
                GameStarted, lambda _: self._on_game_started()
            )
        )

        logging.info("Data collection enabled")

        # if game already started, trigger the game started event to initialize the annotation handler
        if self._model.game is not None:
            self._on_game_started()

    def _open_output_folder(self) -> None:
        """Open the output directory in the system file manager."""
        if not self._output_dir:
            ttk.Messagebox.show_info(
                message="Choose an output directory first.",
                title="Data Collection",
                parent=self._view,
            )
            return

        open_in_file_manager(self._output_dir.resolve())
