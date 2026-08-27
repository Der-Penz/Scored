import tkinter as tk

from PIL import Image

from gui.events.event_channel import EventChannel
from gui.events.event_types import FrameCapturedEvent
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.services.images.http_source import HTTPCaptureSource
from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource
from gui.view.app_view import AppView


SOURCES = [WebCamSource, VideoSource, HTTPCaptureSource]
UPDATE_INTERVAL_MS = int((1 / 30) * 1000)


class CameraFeedController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._left_view = view.left_view
        self._source_view = view.left_view.source_view
        self._latest_frame = None

    def bind_menu(self, menu: tk.Menu) -> None:
        self._feed_visible = tk.BooleanVar(value=True)
        menu.add_checkbutton(
            label="Show Camera Feed",
            variable=self._feed_visible,
            command=self._toggle_feed_visibility,
        )

    def bind_components(self) -> None:
        self._event_channel.subscribe(FrameCapturedEvent, self._on_frame_captured)

    def _on_frame_captured(self, event: FrameCapturedEvent) -> None:
        self._latest_frame = event.frame

    def start(self) -> None:
        self._source_view.after(UPDATE_INTERVAL_MS, self._render_image)

    def _toggle_feed_visibility(self) -> None:
        self._left_view.set_source_visible(self._feed_visible.get())
        self._view.refresh_left_panes()

    def _render_image(self) -> None:
        if self._latest_frame is not None:
            pil = Image.fromarray(self._latest_frame)
            if pil is not None:
                self._source_view.display_image(pil)

        self._source_view.after(UPDATE_INTERVAL_MS, self._render_image)
