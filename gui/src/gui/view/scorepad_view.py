import tkinter as tk
import ttkbootstrap as ttk

from scored_lib.dart.multiplier import Multiplier


class ScorepadView(ttk.Frame):
    """Numeric scorepad with multiplier controls."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, style="Card.TFrame")

        self._create_widgets()

    def _create_widgets(self) -> None:
        rows = 5
        cols = 5
        for r in range(rows):
            self.rowconfigure(r, weight=1)
        for c in range(cols):
            self.columnconfigure(c, weight=1)

        numbers = list(range(1, 21))

        style = {"sticky": "nsew", "padx": 4, "pady": 4}

        self.number_buttons = {}
        for i, n in enumerate(numbers):
            btn = ttk.Button(
                self, text=str(n), command=lambda v=n: self._press_number(v)
            )
            row = i // cols
            col = i % cols
            btn.grid(row=row, column=col, **style)
            self.number_buttons[n] = btn

        self.double_btn = ttk.Button(self, text="D", bootstyle="ghost")
        self.triple_btn = ttk.Button(self, text="T", bootstyle="ghost")
        self.bull25 = ttk.Button(self, text="25", bootstyle="success")
        self.bull50 = ttk.Button(self, text="50", bootstyle="success")
        self.miss = ttk.Button(self, text="Miss", bootstyle="danger")

        self.double_btn.grid(row=4, column=0, **style)
        self.triple_btn.grid(row=4, column=1, **style)
        self.bull25.grid(row=4, column=2, **style)
        self.bull50.grid(row=4, column=3, **style)
        self.miss.grid(row=4, column=4, **style)

    def _prefix_number_btns(self, prefix: str) -> None:
        for n, btn in self.number_buttons.items():
            btn.config(text=f"{prefix}{n}")

    def highlight_multiplier(self, multiplier: Multiplier) -> None:
        if multiplier == Multiplier.SINGLE:
            # change styling
            self.double_btn.config(bootstyle="ghost")
            self.triple_btn.config(bootstyle="ghost")
            self._prefix_number_btns("")
        if multiplier == Multiplier.DOUBLE:
            self.double_btn.config(bootstyle="secondary")
            self.triple_btn.config(bootstyle="ghost")
            self._prefix_number_btns(prefix="D")
        elif multiplier == Multiplier.TRIPLE:
            self.triple_btn.config(bootstyle="secondary")
            self.double_btn.config(bootstyle="ghost")
            self._prefix_number_btns(prefix="T")
