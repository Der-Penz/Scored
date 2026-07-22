import time

import cv2
from gui.services.images.cv2_capture_source import CV2CaptureSource
import numpy as np

class VideoSource(CV2CaptureSource):
    """
    A stream source that reads frames from a video file
    """

    def __init__(self, video_path: str):
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

        Returns
        -------
        np.ndarray | None
            The captured frame, or None if reading failed or finished.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        # Sleep for remaining time to maintain real-time playback rate
        elapsed = time.perf_counter() - self.last_frame_time
        sleep_time = self.frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        return super().read_frame()