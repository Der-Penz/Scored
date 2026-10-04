import tkinter as tk

from PIL import Image

from gui.events.event_channel import EventChannel
from gui.events.event_types import FrameCapturedEvent, PanelRebound
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.services.images.http_source import HTTPCaptureSource
from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource
from gui.view.app_view import AppView
from gui.view.source_view import SourceView

SOURCES = [WebCamSource, VideoSource, HTTPCaptureSource]
UPDATE_INTERVAL_MS = int((1 / 30) * 1000)


class CameraFeedController(BaseController):
    """
    Renders the captured frames into the camera feed panel.

    The render loop and the frame subscription only run while the panel is
    actually on screen; the layout controller drives that through
    :meth:`set_active`.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._source_view: SourceView | None = None
        self._latest_frame = None
        self.capture_event_subscription = None
        self._active = False
        self._after_id: str | None = None

    def bind_menu(self, menu: tk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        self._event_channel.subscribe(PanelRebound, self._on_panel_rebound)

    def _on_frame_captured(self, event: FrameCapturedEvent) -> None:
        self._latest_frame = event.frame

    def start(self) -> None:
        self._source_view = self._view.source_view
        if self._active:
            self._start_rendering()

    # -- visibility -------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Start or stop rendering as the camera feed panel is shown or hidden."""
        if active == self._active:
            return

        self._active = active
        if active:
            self._subscribe()
            self._start_rendering()
        else:
            self._stop_rendering()
            self._unsubscribe()

    def _subscribe(self) -> None:
        if self.capture_event_subscription is None:
            self.capture_event_subscription = self._event_channel.subscribe(
                FrameCapturedEvent, self._on_frame_captured
            )

    def _unsubscribe(self) -> None:
        if self.capture_event_subscription is not None:
            self.capture_event_subscription.cancel()
            self.capture_event_subscription = None

    # -- render loop ------------------------------------------------------

    def _start_rendering(self) -> None:
        if self._after_id is None:
            self._after_id = self._view.after(UPDATE_INTERVAL_MS, self._render_image)

    def _stop_rendering(self) -> None:
        if self._after_id is not None:
            try:
                self._view.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _render_image(self) -> None:
        self._after_id = None

        if self._latest_frame is not None and self._source_view is not None:
            self._source_view.display_image(Image.fromarray(self._latest_frame))

        if self._active:
            self._after_id = self._view.after(UPDATE_INTERVAL_MS, self._render_image)

    # -- panel lifecycle --------------------------------------------------

    def _on_panel_rebound(self, event: PanelRebound) -> None:
        """Re-acquire the feed panel after it was rebuilt in a new column."""
        if event.panel is not Panel.SOURCE:
            return

        self._stop_rendering()
        self._source_view = self._view.source_view
        if self._active:
            self._start_rendering()
