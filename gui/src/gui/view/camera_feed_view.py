import tkinter as tk

from PIL import Image, ImageTk

from scored_lib.dart.dart_throw import DartThrow


class CameraFeedView(tk.Frame):
    """View for the live camera or source feed."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, bg="red")
        self.image_label = tk.Label(self)
        self.image_label.pack(fill="both", expand=True)
        self._photo_image = None

    def addDartThrow(self, _dart_throw: DartThrow) -> None:
        pass

    def resetDartThrow(self) -> None:
        pass

    def display_image(self, pil_image: Image.Image) -> None:
        """Display a PIL image in the feed view."""
        if pil_image is None:
            self.image_label.config(image="")
            self._photo_image = None
            return

        width = self.image_label.winfo_width() or pil_image.width
        height = self.image_label.winfo_height() or pil_image.height
        image = pil_image.copy()
        image.thumbnail((width, height), Image.LANCZOS)

        self._photo_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self._photo_image)