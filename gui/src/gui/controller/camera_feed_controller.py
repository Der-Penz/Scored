import threading
import time
import tkinter as tk
from tkinter import filedialog, simpledialog

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
    def __init__(self, view: AppView, model: AppModel):
        super().__init__(view, model)
        self._view = view
        self._model = model

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
        pass

    def _toggle_feed_visibility(self) -> None:
        self._view.set_image_visible(self._feed_visible.get())
