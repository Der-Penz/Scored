from abc import ABC

import cv2
from gui.services.images.source import ImageSource
import numpy as np


class CV2CaptureSource(ImageSource, ABC):
    """
    A stream source that uses OpenCV to capture frames from a video source (e.g. webcam, video file)
    """

    def __init__(self, source: int | str):
        """
        Init the CV2CaptureSource

        Parameters
        ----------
        source : int | str
            The video source (e.g. webcam index or video file path)
        """
        super().__init__()
        self.source = source
        self.cap: cv2.VideoCapture = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)

    def read_frame(self) -> np.ndarray:
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def close(self):
        if self.cap is None:
            return

        cap_to_close = self.cap
        self.cap = None
        cap_to_close.release()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
