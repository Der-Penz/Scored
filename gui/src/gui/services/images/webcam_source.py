from gui.services.images.cv2_capture_source import CV2CaptureSource


class WebCamSource(CV2CaptureSource):
    """
    A stream source that reads frames from a webcam
    """

    def __init__(self, cam_index: int = 0):
        """
        Init the webcam stream source

        Parameters
        ----------
        cam_index : int, optional
            The index of the webcam to use (default is 0)
        """
        super().__init__(cam_index)

    @classmethod
    def get_name(cls) -> str:
        """
        Returns the name of the source class for display in the UI.

        Returns
        -------
        str
            The name of the class.
        """
        return "Webcam"
