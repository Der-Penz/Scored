import threading
import time
from tkinter import filedialog, simpledialog

from gui.events.event_channel import EventChannel
import ttkbootstrap as ttk

from gui.events.event_types import FrameCapturedEvent
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.services.images.http_source import HTTPCaptureSource
from gui.services.images.source import ImageSource
from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource
from gui.view.app_view import AppView

SOURCES = [
    (WebCamSource, "Ctrl+W"),
    (VideoSource, "Ctrl+V"),
    (HTTPCaptureSource, "Ctrl+H"),
]
UPDATE_INTERVAL_MS = int((1 / 30) * 1000)  # Update interval for the UI in milliseconds


class SourceController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._running = False
        self._thread = None
        self._current_source: ImageSource | None = None

    def bind_menu(self, menu: ttk.Menu) -> None:
        for source, accel in SOURCES:
            menu.add_command(
                label=source.get_name(),
                command=lambda src=source.get_name(): self.on_select_source(src),
                accelerator=accel,
            )

        menu.add_separator()
        menu.add_command(label="Stop Stream", command=self.stop)

    def bind_components(self):
        self._view.master.bind_all(
            "<Control-w>", lambda event: self.on_select_source(WebCamSource.get_name())
        )
        self._view.master.bind_all(
            "<Control-h>",
            lambda event: self.on_select_source(HTTPCaptureSource.get_name()),
        )
        self._view.master.bind_all(
            "<Control-v>", lambda event: self.on_select_source(VideoSource.get_name())
        )

    def start(self):
        pass

    def set_source_by_value(self, value: str):
        # check for int or http or path
        if value.isdigit():
            source = WebCamSource(int(value))
        elif value.startswith("http"):
            source = HTTPCaptureSource(value)
        else:
            source = VideoSource(value)

        self.set_source(source)

    def set_source(self, source: ImageSource) -> None:
        """
        Stops the current stream and starts a new one with the given source.

        Parameters
        ----------
        source : ImageSource
            The initialized image source object to stream from.
        """
        self.stop()
        self._current_source = source
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the source and releases resources."""
        self._running = False
        if self._current_source is not None:
            self._current_source.close()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._thread = None

    def _update(self) -> None:
        """Reads frames continuously from the capture device."""

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
                        self.event_channel.post(FrameCapturedEvent(processed_frame))
                else:
                    time.sleep(0.01)
        except Exception:
            # Prevent background thread crash from corrupting state
            pass
        finally:
            if self._current_source is not None:
                self._current_source.close()
            self._running = False

    def on_select_source(self, source_name: str) -> None:
        """
        Handle source selection from the UI. Prompts the user for any required
        parameters (camera index, file path, URL) and then starts the stream.
        """
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
                for name, cls in self.source_manager.get_sources():
                    if name == source_name:
                        source = cls()
                        break
                else:
                    raise ValueError(f"Unknown source: {source_name}")

            self.set_source(source)

        except Exception as _:
            self.set_source(None)
