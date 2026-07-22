from abc import ABC
from queue import Queue

import cv2
from gui.services.images.source import ImageSource
import numpy as np


class CV2CaptureSource(ImageSource, ABC):
    """
    A stream source that uses OpenCV to capture frames from a video source (e.g. webcam, video file)
    """

    def __init__(self, output_queue: Queue, source: int | str):
        """
        Init the CV2CaptureSource

        Parameters
        ----------
        output_queue : Queue
            The queue to put the frames into
        source : int | str
            The video source (e.g. webcam index or video file path)
        """
        super().__init__(output_queue)
        self.source = source
        self.cap: cv2.VideoCapture = None

    def _open(self):
        self.cap = cv2.VideoCapture(self.source)

    def _read_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame from source {self.source}")
        return frame

    def _close(self):
        if self.cap:
            self.cap.release()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame
