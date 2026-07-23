from gui.controller.source_manager import SourceManager
from gui.model.model import AppModel
from gui.services.images.http_source import HTTPCaptureSource
from gui.view.app_view import AppView
from gui.model.args import AppConfig

from gui.services.images.video_source import VideoSource
from gui.services.images.webcam_source import WebCamSource

from tkinter import filedialog, simpledialog
from PIL import Image

UPDATE_INTERVAL_MS = int((1 / 30) * 1000)  # Update interval for the UI in milliseconds


class AppController:
    """
    Coordinates the application state, user interface, and background processes.
    """

    def __init__(self, model: AppModel, view: AppView, config: AppConfig):
        self.model = model
        self.view = view
        self.config = config
        self.source_manager = SourceManager()
        self._bind_events()

    def _bind_events(self) -> None:
        """
        Bind UI callbacks to controller methods.
        """
        for source in self.source_manager.get_sources():
            self.view.source_menu.add_command(
                label=source.get_name(),
                command=lambda src=source.get_name(): self._on_select_source(src),
            )

        self.view.source_menu.add_separator()
        self.view.source_menu.add_command(
            label="Stop Stream", command=self.source_manager.stop
        )

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
                source = VideoSource(path)

            elif source_name == WebCamSource.get_name():
                idx = simpledialog.askinteger(
                    "Camera index", "Enter camera index (0,1,...):", minvalue=0
                )
                if idx is None:
                    return
                source = WebCamSource(idx)

            elif source_name == HTTPCaptureSource.get_name():
                url = simpledialog.askstring(
                    "Stream URL", "Enter HTTP stream URL:", initialvalue="http://"
                )
                if not url:
                    return
                source = HTTPCaptureSource(url)

            else:
                for name, cls in self.source_manager.get_sources():
                    if name == source_name:
                        source = cls()
                        break
                else:
                    raise ValueError(f"Unknown source: {source_name}")

            self.source_manager.set_source(source)

        except Exception as exc:
            # print the exception to the console for debugging
            import traceback

            traceback.print_exc()
            self.source_manager.set_source(None)

    def start(self):
        if self.config.source is not None:
            self.source_manager.set_source_by_value(self.config.source)
        self.view.after(UPDATE_INTERVAL_MS, self._render)
        self.view.mainloop()

    def _render(self) -> None:
        frame = self.source_manager.get_frame()
        if frame is not None:
            pil = Image.fromarray(frame)
            if pil is not None:
                self.view.display_image(pil)

        self.view.after(UPDATE_INTERVAL_MS, self._render)
