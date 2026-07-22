from queue import Queue

from gui.services.images.cv2_capture_source import CV2CaptureSource



class VideoSource(CV2CaptureSource):
    """
    A stream source that reads frames from a video file
    """

    def __init__(self, output_queue: Queue, video_path: str):
        """
        Init the video source

        Parameters
        ----------
        output_queue : Queue
            The queue to put the frames into
        video_path : str
            The path to the video file to read from
        """
        super().__init__(output_queue, video_path)

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