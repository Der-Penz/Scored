import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk as _ttk
from typing import Callable

import ttkbootstrap as ttk

DIM_FILL = "#000000"
DIM_ALPHA = 0.5
NEXT_STYLE = "TurnEnd.Next.TButton"


class TurnEndOverlay(ttk.Frame):
    """
    Modal overlay that blocks the scorepad after a finished turn.

    A snapshot of the scoreboard, dimmed by 50% black, covers the scorepad so
    no buttons can be pressed. A full-width confirmation button at the bottom
    advances to the next player; clicking anywhere else on the overlay undoes
    the last throw.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._on_next: Callable[[], None] | None = None
        self._on_undo: Callable[[], None] | None = None
        self._background_photo = None
        self._rect_id = None
        self._image_id = None

        self.canvas = tk.Canvas(self, highlightthickness=0, bg="black")
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", lambda _e: self._trigger_undo())

        self._create_button_style()
        self.next_button = _ttk.Button(
            self,
            text="Next",
            style=NEXT_STYLE,
            command=self._trigger_next,
        )
        self.next_button.place(relx=0.5, rely=1.0, anchor="s", relwidth=1.0, y=-8)

        self.hide()

    def _create_button_style(self) -> None:
        """Build a large success-colored button style unique to this overlay."""
        style = ttk.Style()
        source = "success.TButton"
        style.configure(
            NEXT_STYLE,
            **{name: style.configure(source, name) for name in style.configure(source)},
        )
        style.map(NEXT_STYLE, **{name: style.map(source, name) for name in style.map(source)})
        style.configure(
            NEXT_STYLE,
            font=tkfont.Font(size=20, weight="bold"),
            padding=(18, 12),
        )

    def bind_callbacks(self, on_next: Callable[[], None], on_undo: Callable[[], None]) -> None:
        """Register the callbacks for the next/undo actions."""
        self._on_next = on_next
        self._on_undo = on_undo

    def show(self) -> None:
        """Capture the dimmed background and cover the whole scoreboard."""
        self._capture_background()
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()
        self.focus_force()
        self.next_button.focus_set()
        self._bind_keys()

    def hide(self) -> None:
        """Remove the overlay from the scoreboard again."""
        self.place_forget()
        self._unbind_keys()

    def is_visible(self) -> bool:
        return bool(self.winfo_ismapped())

    def _bind_keys(self) -> None:
        for widget in (self, self.canvas, self.next_button):
            widget.bind("<Return>", lambda _e: self._trigger_next())
            widget.bind("<Escape>", lambda _e: self._trigger_undo())

    def _unbind_keys(self) -> None:
        for widget in (self, self.canvas, self.next_button):
            widget.unbind("<Return>")
            widget.unbind("<Escape>")

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        self._redraw_background()

    def _capture_background(self) -> None:
        """Snapshot the scoreboard behind the overlay and dim it by 50% black."""
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        if width <= 1 or height <= 1:
            self._background_photo = None
            return

        try:
            from PIL import Image, ImageGrab, ImageTk

            x = self.master.winfo_rootx()
            y = self.master.winfo_rooty()
            image = ImageGrab.grab((x, y, x + width, y + height)).convert("RGB")
            black = Image.new("RGB", image.size, (0, 0, 0))
            self._background_photo = ImageTk.PhotoImage(Image.blend(image, black, DIM_ALPHA))
        except Exception:
            self._background_photo = None
        self._redraw_background()

    def _redraw_background(self) -> None:
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        if self._image_id is not None:
            self.canvas.delete(self._image_id)
            self._image_id = None

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        if self._background_photo is not None:
            self._image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self._background_photo
            )
        else:
            self._rect_id = self.canvas.create_rectangle(
                0,
                0,
                width,
                height,
                fill=DIM_FILL,
                outline="",
            )

    def _trigger_next(self) -> None:
        if self._on_next is not None:
            self._on_next()

    def _trigger_undo(self) -> None:
        if self._on_undo is not None:
            self._on_undo()
