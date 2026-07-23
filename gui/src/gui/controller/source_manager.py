import threading
import time

from gui.services.images.http_source import HTTPCaptureSource
from gui.services.images.source import ImageSource
from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource
import numpy as np


class SourceManager:
    def __init__(self):
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._current_source: ImageSource | None = None

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
                    with self._lock:
                        self._latest_frame = processed_frame
                else:
                    time.sleep(0.01)
        except Exception:
            # Prevent background thread crash from corrupting state
            pass
        finally:
            if self._current_source is not None:
                self._current_source.close()
            self._running = False

    def get_frame(self) -> np.ndarray | None:
        """
        Retrieves the most recently captured frame safely.

        Returns
        -------
        np.ndarray | None
            The current frame as a numpy array, or None if no frame is available.
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_sources(self) -> list[type[ImageSource]]:
        """
        Returns a list of available image sources.

        Returns
        -------
        list[type[ImageSource]]
            List of available image sources.
        """
        return [WebCamSource, VideoSource, HTTPCaptureSource]
