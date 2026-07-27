import threading
import time
import tkinter as tk
from tkinter import filedialog, simpledialog

from gui.controller.source_controller import SourceController
import numpy as np
from PIL import Image

from gui.protocols.controller_protocol import ControllerProtocol
from gui.model.model import AppModel
from gui.services.images.http_source import HTTPCaptureSource
from gui.services.images.source import ImageSource
from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource
from gui.view.app_view import AppView


SOURCES = [WebCamSource, VideoSource, HTTPCaptureSource]
UPDATE_INTERVAL_MS = int((1 / 30) * 1000)


class CameraFeedController(ControllerProtocol):
    def __init__(
        self, view: AppView, model: AppModel, source_controller: SourceController
    ):
        super().__init__(view, model)
        self._view = view
        self._model = model
        self._left_view = view.left_view
        self._source_view = view.left_view.source_view
        self._source_controller = source_controller

    def bind_menu(self, menu_bar: tk.Menu) -> None:
        self._feed_visible = tk.BooleanVar(value=True)
        menu_bar.add_checkbutton(
            label="Show Camera Feed",
            variable=self._feed_visible,
            command=self._toggle_feed_visibility,
        )

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        self._source_view.after(UPDATE_INTERVAL_MS, self._render_image)

    def _toggle_feed_visibility(self) -> None:
        self._left_view.set_source_visible(self._feed_visible.get())
        self._view.refresh_left_panes()

    def _render_image(self) -> None:
        frame = self._source_controller.get_frame()
        if frame is not None:
            pil = Image.fromarray(frame)
            if pil is not None:
                self._source_view.display_image(pil)

        self._source_view.after(UPDATE_INTERVAL_MS, self._render_image)
