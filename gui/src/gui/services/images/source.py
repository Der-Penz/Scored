from abc import ABC, abstractmethod
from multiprocessing import Queue
import time

import numpy as np


class ImageSource(ABC):
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

    def run(self) -> None:
        """
        Starts the blocking capture loop.
        """
        self._running = True
        self._open()
        
        try:
            while self._running:
                frame = self._read_frame()
                
                if frame is None:
                    break
                    
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None:
                    self.output_queue.put(processed_frame)
                    
                # Yield a tiny amount of time to prevent maxing out the CPU core
                time.sleep(1 / 30)  # Assuming a target of 30 FPS
        finally:
            self._close()

    def stop(self):
        """
        stops the stream source
        """
        self._running = False

    @classmethod
    def get_name(cls) -> str:
        """
        Returns the name of the source class for display in the UI.

        Returns
        -------
        str
            The name of the class.
        """
        return cls.__name__
    
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