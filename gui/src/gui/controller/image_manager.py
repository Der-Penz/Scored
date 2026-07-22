import multiprocessing as mp

from gui.services.images.htpp_source import HTTPCaptureSource
from gui.services.images.source import ImageSource
from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource

class StreamManager:
    """
    Manages the lifecycle of the multiprocessing image stream.
    """

    def __init__(self):
        """
        Init the stream manager.
        
        """
        self._frame_queue = mp.Queue()
        self._process: mp.Process | None = None
        self.current_source: ImageSource | None = None

    def set_source(self, source: ImageSource) -> None:
        """
        Stops the current stream and starts a new one with the given source.

        Parameters
        ----------
        source : ImageSource
            The initialized image source object to stream from.
        """
        self.stop()
        self.current_source = source.get_name() if source else ""

        self._process = mp.Process(
            target=source.run,
            daemon=True
        )
        self._process.start()

    def stop(self) -> None:
        """
        Terminates the current stream process safely.
        """
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join()
            self._process = None
        self.current_source = None
            
    def get_queue(self) -> mp.Queue:
        """
        Returns the multiprocessing queue used for frame transfer.

        Returns
        -------
        mp.Queue
            The queue frames are put into
        """
        return self._frame_queue
    
    def get_sources(self) -> list[tuple[str, ImageSource]]:
        """
        Returns a list of available image sources.

        Returns
        -------
        list[tuple[str, ImageSource]]
            List of available image sources.
        """
        return [WebCamSource, VideoSource, HTTPCaptureSource]