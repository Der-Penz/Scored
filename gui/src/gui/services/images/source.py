from abc import ABC, abstractmethod

import numpy as np


class ImageSource(ABC):
    """
    A base class for streaming frames to the application from different sources
    """

    def __init__(self):
        """
        Init the stream source
        """

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
    def open(self):
        """
        open the stream source
        """
        pass

    @abstractmethod
    def read_frame(self) -> np.ndarray:
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
    def close(self):
        """
        close up any resources
        """
        pass
