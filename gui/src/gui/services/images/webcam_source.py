from queue import Queue

from gui.services.images.cv2_capture_source import CV2CaptureSource



class WebCamSource(CV2CaptureSource):
    """
    A stream source that reads frames from a webcam
    """

    def __init__(self, output_queue: Queue, cam_index: int = 0):
        """
        Init the webcam stream source

        Parameters
        ----------
        output_queue : Queue
            The queue to put the frames into
        cam_index : int, optional
            The index of the webcam to use (default is 0)
        """
        super().__init__(output_queue, cam_index)
        
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
