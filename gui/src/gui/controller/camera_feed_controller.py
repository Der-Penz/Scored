import threading
import time
import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image

from gui.controller.controller import Controller
from gui.model.model import AppModel
from gui.services.images.http_source import HTTPCaptureSource
from gui.services.images.source import ImageSource
from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource
from gui.view.app_view import AppView


SOURCES = [WebCamSource, VideoSource, HTTPCaptureSource]
UPDATE_INTERVAL_MS = int((1 / 30) * 1000)


class CameraFeedController(Controller):
    def __init__(self, view: AppView, model: AppModel, view_menu, source_menu):
        super().__init__(view, model)
        self._view = view
        self._model = model
        self._view_menu = view_menu
        self._source_menu = source_menu
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._current_source: ImageSource | None = None

    def bind_menu(self) -> None:
        self._feed_visible = tk.BooleanVar(value=True)
        self._view_menu.add_checkbutton(
            label="Show Camera Feed",
            variable=self._feed_visible,
            command=self._toggle_feed_visibility,
        )

        for source in SOURCES:
            self._source_menu.add_command(
                label=source.get_name(),
                command=lambda src=source.get_name(): self.on_select_source(src),
            )

        self._source_menu.add_separator()
        self._source_menu.add_command(label="Stop Stream", command=self.stop)

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        self._view.after(UPDATE_INTERVAL_MS, self._rerender_images)

    def _toggle_feed_visibility(self) -> None:
        self._view.set_image_visible(self._feed_visible.get())

    def set_source_by_value(self, value: str):
        if value.isdigit():
            source = WebCamSource(int(value))
        elif value.startswith("http"):
            source = HTTPCaptureSource(value)
        else:
            source = VideoSource(value)

        self.set_source(source)

    def set_source(self, source: ImageSource) -> None:
        self.stop()
        self._current_source = source
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._current_source is not None:
            self._current_source.close()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._thread = None

    def _update(self) -> None:
        if self._current_source is None:
            return

        self._current_source.open()
        self._running = True

        try:
            while self._running:
                frame = self._current_source.read_frame()

                if frame is None or not self._running:
                    break

                processed_frame = self._current_source.process_frame(frame)
                if processed_frame is not None:
                    with self._lock:
                        self._latest_frame = processed_frame
                else:
                    time.sleep(0.01)
        except Exception:
            pass
        finally:
            if self._current_source is not None:
                self._current_source.close()
            self._running = False

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def on_select_source(self, source_name: str) -> None:
        try:
            if source_name == VideoSource.get_name():
                path = filedialog.askopenfilename(title="Open video file")
                if not path:
                    return
                source = VideoSource(path)

            elif source_name == WebCamSource.get_name():
                idx = simpledialog.askinteger(
                    "Camera index", "Enter camera index (0,1,...):", minvalue=0
                )
                if idx is None:
                    return
                source = WebCamSource(idx)

            elif source_name == HTTPCaptureSource.get_name():
                url = simpledialog.askstring(
                    "Stream URL", "Enter HTTP stream URL:", initialvalue="http://"
                )
                if not url:
                    return
                source = HTTPCaptureSource(url)

            else:
                for name, cls in [(source.get_name(), source) for source in SOURCES]:
                    if name == source_name:
                        source = cls()
                        break
                else:
                    raise ValueError(f"Unknown source: {source_name}")

            self.set_source(source)

        except Exception:
            self.set_source(None)

    def _rerender_images(self) -> None:
        frame = self.get_frame()
        if frame is not None:
            pil = Image.fromarray(frame)
            if pil is not None:
                self._view.display_image(pil)

        self._view.after(UPDATE_INTERVAL_MS, self._rerender_images)