import tkinter as tk
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier
import ttkbootstrap as ttk
from tkinter import font as tkfont
from typing import Callable, Sequence

EMPTY_THROW = "➜"


class TurnView(ttk.Frame):
    """Turn view with dart throws"""

    def __init__(self, master: tk.Misc):
        super().__init__(master, style="Card.TFrame", padding=10)

        self._throw_value_labels: list[ttk.Label] = []
        self._throw_slots: list[ttk.Labelframe] = []
        self._active_throw: int | None = 1

        self._create_widgets()
        self.reset_throws()

    def _create_widgets(self) -> None:
        """Create and layout the widgets for the view."""
        self.columnconfigure(0, weight=1)

        throws_container = ttk.Frame(self)
        throws_container.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        throws_container.columnconfigure((0, 1, 2), weight=1)

        for idx in range(3):
            slot = ttk.Labelframe(throws_container, text=f"Throw {idx + 1}")
            slot.grid(row=0, column=idx, sticky="nsew", padx=4)
            slot.columnconfigure(0, weight=1)

            value = ttk.Label(
                slot,
                text="",
                anchor="center",
                padding=(4, 6),
                font=tkfont.Font(size=16, weight="bold"),
                bootstyle="secondary",
            )
            value.grid(row=1, column=0, sticky="ew", pady=0)

            self._throw_slots.append(slot)
            self._throw_value_labels.append(value)

    def reset_throws(self) -> None:
        for label in self._throw_value_labels:
            label.config(text=EMPTY_THROW)
        self.highlight_throw(1)

    def set_throw(self, throw_number: int, score_value: str) -> None:
        self._throw_value_labels[throw_number - 1].config(text=score_value)

        if throw_number < 3:
            self.highlight_throw(throw_number + 1)
        else:
            self.highlight_throw(None)

    def set_ghostpath(self, path: tuple[DartThrow, ...]) -> None:
        for label, throw in zip(self._throw_value_labels, path):
            label.config(
                text=throw.short_label,
                font=tkfont.Font(size=16, weight="normal", slant="italic"),
                foreground="#464b44",
            )

    def highlight_throw(self, throw_number: int | None) -> None:
        for idx, slot in enumerate(self._throw_slots):
            if throw_number is not None and idx + 1 == throw_number:
                slot.config(bootstyle="info")
            else:
                slot.config(bootstyle="secondary")
