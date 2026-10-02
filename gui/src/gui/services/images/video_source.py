import logging
import time

import cv2
import numpy as np
from gui.services.images.cv2_capture_source import CV2CaptureSource


class VideoSource(CV2CaptureSource):
    """
    A stream source that reads frames from a video file
    """

    def __init__(self, video_path: str, loop: bool = True):
        """
        Init the video source

        Parameters
        ----------
        video_path : str
            The path to the video file to read from
        """
        super().__init__(video_path)
        self.fps = 30.0
        self.frame_delay = 1.0 / self.fps
        self.last_frame_time = 0.0
        self.loop = loop

    @classmethod
    def get_name(cls) -> str:
        """
        Returns the name of the source class for display in the UI.

        Returns
        -------
        str
            The name of the class.
        """
        return "Video File"

    def open(self) -> None:
        """Open the video file and retrieve its native FPS."""
        self.cap = cv2.VideoCapture(self.source)
        detected_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if detected_fps > 0:
            self.fps = detected_fps

        self.frame_delay = 1.0 / self.fps
        self.last_frame_time = time.perf_counter()

    def read_frame(self) -> np.ndarray | None:
        """
        Read a frame from the video file, maintaining natural FPS timing.

        When the end of the video is reached the source rewinds back to the
        first frame so playback starts over (unless looping is disabled).

        Returns
        -------
        np.ndarray | None
            The captured frame, or None if reading failed or finished.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        elapsed = time.perf_counter() - self.last_frame_time
        sleep_time = self.frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(
                sleep_time * 0.9
            )  # Sleep for 90% of the remaining time to avoid overshooting

        self.last_frame_time = time.perf_counter()

        frame = super().read_frame()
        if frame is None:
            if not self.loop:
                return None

            if not self._rewind():
                return None
            logging.debug("Video looped back to the start")
            self.last_frame_time = time.perf_counter()
            frame = super().read_frame()
            if frame is None:
                # Nothing could be read after rewinding, stop instead of looping
                # forever on an empty/unreadable video.
                self.loop = False

        return frame

    def _rewind(self) -> bool:
        """
        Seek back to the start of the video so it can be replayed.

        Returns
        -------
        bool
            True if the capture was successfully rewound, False otherwise.
        """
        if self.cap is None or not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return not self.cap.get(cv2.CAP_PROP_POS_FRAMES) > 0
