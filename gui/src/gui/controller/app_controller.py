from gui.controller.image_manager import StreamManager
from gui.model.model import AppModel
from gui.services.images.htpp_source import HTTPCaptureSource
from gui.view.app_view import AppView
from gui.model.args import AppConfig

from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource

from tkinter import filedialog, simpledialog, messagebox
from PIL import Image
import numpy as np
import cv2
import queue as _queue


class AppController():
    """
    Coordinates the application state, user interface, and background processes.
    """
    def __init__(self, model : AppModel, view : AppView, config: AppConfig):
        self.model = model
        self.view = view
        self.config = config
        self.stream_manager = StreamManager()
        self._bind_events()
        
    def _bind_events(self) -> None:
        """
        Bind UI callbacks to controller methods.
        """
        for source in self.stream_manager.get_sources():
            self.view.source_menu.add_command(
                label=source.get_name(),
                command=lambda src=source.get_name(): self._on_select_source(src)
            )

        self.view.source_menu.add_separator()
        self.view.source_menu.add_command(label="Stop Stream", command=self.stream_manager.stop)

    def _on_select_source(self, source_name: str) -> None:
        """
        Handle source selection from the UI. Prompts the user for any required
        parameters (camera index, file path, URL) and then starts the stream.
        """
        try:
            if source_name == VideoSource.get_name():
                path = filedialog.askopenfilename(title="Open video file")
                if not path:
                    return
                source = VideoSource(self.stream_manager.get_queue(), path)

            elif source_name == WebCamSource.get_name():
                idx = simpledialog.askinteger("Camera index", "Enter camera index (0,1,...):", minvalue=0)
                if idx is None:
                    return
                source = WebCamSource(self.stream_manager.get_queue(), idx)

            elif source_name == HTTPCaptureSource.get_name():
                url = simpledialog.askstring("Stream URL", "Enter HTTP stream URL:", initialvalue="http://")
                if not url:
                    return
                source = HTTPCaptureSource(self.stream_manager.get_queue(), url)

            else:
                for name, cls in self.stream_manager.get_sources():
                    if name == source_name:
                        source = cls(self.stream_manager.get_queue())
                        break
                else:
                    raise ValueError(f"Unknown source: {source_name}")

            self.stream_manager.set_source(source)

        except Exception as exc:
            # print the exception to the console for debugging
            import traceback
            traceback.print_exc()
            messagebox.showerror("Failed to start source", str(exc))

    def start(self):
        # start polling the frame queue and then enter the Tk mainloop
        self.view.after(30, self._poll_queue)
        self.view.mainloop()

    def _poll_queue(self) -> None:
        """Poll the stream manager queue for frames and display them in the view."""
        q = self.stream_manager.get_queue()
        try:
            while True:
                frame = q.get_nowait()
                if frame is None:
                    # stream ended
                    self.view.display_image(None)
                    continue

                # Convert numpy BGR frame to PIL Image (RGB)
                try:
                    if isinstance(frame, np.ndarray):
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                    else:
                        # if it's already a PIL image or other format, try to construct
                        pil = Image.fromarray(frame)
                except Exception:
                    # fallback: ignore this frame
                    pil = None

                if pil is not None:
                    self.view.display_image(pil)

        except _queue.Empty:
            # no frames available right now
            pass
        finally:
            # schedule next poll
            try:
                self.view.after(30, self._poll_queue)
            except Exception:
                # view may have been destroyed
                pass