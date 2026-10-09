import numpy as np
import ttkbootstrap as ttk

import gui.events.event_types as events
from gui.events.event_channel import EventChannel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class ImageCollectionController(BaseController):
    """
    Collect frames used for annotating
    """
    
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._current_frame: np.ndarray | None = None

    def bind_menu(self, _: ttk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        self._event_channel.subscribe(events.FrameCapturedEvent, self._save_frame)
        self._event_channel.subscribe(events.ThrowRecorded, self._on_throw_captured)
        self._event_channel.subscribe(events.ThrowRemoved, self._on_throw_removed)
        self._event_channel.subscribe(events.TurnChanged, self._on_turn_changed)

    def _save_frame(self, event: events.FrameCapturedEvent) -> None:
        if event.frame is not None:
            self._current_frame = event.frame

    def _on_throw_captured(self, _: events.ThrowRecorded) -> None:
        if self._current_frame is not None:
            self._model.frame_snapshots.append(self._current_frame)
            self._current_frame = None

    def _on_throw_removed(self, _: events.ThrowRemoved) -> None:
        if self._model.frame_snapshots:
            self._model.frame_snapshots.pop()

    def _on_turn_changed(self, _: events.TurnChanged) -> None:
        self._current_frame = None
        self._model.frame_snapshots.clear()
