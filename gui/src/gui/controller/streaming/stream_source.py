from abc import ABC, abstractmethod
from queue import Queue
import threading

import numpy as np


class StreamSource(ABC):
    """
    A base class for streaming frames to the application from different sources
    The stream source will run in a separate thread and put frames into the output queue for processing by the application.
    """

    def __init__(self, output_queue: Queue):
        """
        Init the stream source

        Parameters
        ----------
        output_queue : Queue
            queue to put the frames into
        """
        self.output_queue = output_queue
        self._running: bool = False
        self._thread: threading.Thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        stops the stream source
        """
        self._running = False
        if self._thread:
            self._thread.join()

    def _run(self):
        """Internal thread loop"""
        self._open()
        try:
            while self._running:
                frame = self._read_frame()
                process = self.process_frame(frame)
                if process is not None:
                    self.output_queue.put(process)
        finally:
            self._close()

    @abstractmethod
    def _open(self):
        """
        open the stream source
        """
        pass

    @abstractmethod
    def _read_frame(self) -> np.ndarray:
        """
        Read a frame from the source

        Returns
        -------
        np.ndarray
            The frame read from the source
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the frame before putting it into the queue

        Parameters
        ----------
        frame : np.ndarray
            The frame to process

        Returns
        -------
        np.ndarray
            The processed frame
        """
        pass

    @abstractmethod
    def _close(self):
        """
        close up any resources
        """
        pass
