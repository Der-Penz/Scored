import cv2
from gui.services.images.cv2_capture_source import CV2CaptureSource


class HTTPCaptureSource(CV2CaptureSource):
    """
    A stream source that uses OpenCV to capture frames from an HTTP or HTTPS video stream.
    """

    def __init__(self, url: str):
        """
        Initialize the HTTPCaptureSource.

        Parameters
        ----------
        url : str
            The HTTP or HTTPS URL of the video stream (e.g., an IP camera feed)
        """
        super().__init__(url)

    def open(self) -> None:
        """
        Open the HTTP video stream using the FFMPEG backend.
        """
        stream_url = self.source

        assert isinstance(stream_url, str), "The source URL must be a string."

        if not stream_url.startswith(("http://", "https://")):
            stream_url = "http://" + stream_url

        self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open HTTP stream at {stream_url}")

    @classmethod
    def get_name(cls) -> str:
        """
        Returns the name of the source class for display in the UI.

        Returns
        -------
        str
            The name of the class.
        """
        return "HTTP Stream"
