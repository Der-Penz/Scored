import tkinter as tk

from scored_lib.dart.dart_throw import DartThrow

from gui.view.drawing.dartboard import draw_dartboard


class DartboardView(tk.Frame):
    """Canvas-backed view for drawing the digital dartboard."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, bg="black")
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.after_idle(self.redraw)

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        self.redraw()

    def redraw(self) -> None:
        """Redraw the dartboard background using the current canvas size."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        size = min(width, height)

        if size <= 1:
            return

        self.canvas.delete("all")
        draw_dartboard(self.canvas, size)

    def addDartThrow(self, _dart_throw: DartThrow) -> None:
        pass

    def resetDartThrow(self) -> None:
        pass