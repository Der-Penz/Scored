from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont

import ttkbootstrap as ttk

from gui.helper import center_dialog

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
POLL_INTERVAL_MS = 100
MAX_LINES = 5000

# Keys that may not change the content of the log area.
NAVIGATION_KEYS = frozenset(
    {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next", "Escape"}
)
CONTROL_KEYS = frozenset({"c", "a", "f"})
CONTROL_MASK = 0x0004


class LoggingView(ttk.Toplevel):
    """
    Scrollable window showing the log records emitted by the application.

    The view only displays text: it formats nothing and reads no files. The
    feeding controller calls :meth:`append_lines` for the records it received
    and :meth:`start_polling` to have the view pull them periodically, which
    keeps all Tk calls on the main thread.
    """

    def __init__(
        self,
        master: tk.Misc,
        level: str = "INFO",
        on_level_change: Callable[[str], None] | None = None,
        on_open_folder: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Application Logs")
        self.transient(master)
        self.geometry("820x460")
        self.minsize(520, 300)
        center_dialog(self, self.master)

        self._on_level_change = on_level_change
        self._on_open_folder = on_open_folder
        self._autoscroll = tk.BooleanVar(value=True)
        self._poll: Callable[[], list[str]] | None = None
        self._after_id: str | None = None

        self.create_widgets(level)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def create_widgets(self, level: str) -> None:
        """Build the toolbar and the scrollable log area."""
        toolbar = ttk.Frame(self, padding=(8, 8, 8, 0))
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Level:").pack(side="left")
        self._level_box = ttk.Combobox(
            toolbar, values=LEVELS, width=10, state="readonly"
        )
        self._level_box.set(level if level in LEVELS else "INFO")
        self._level_box.pack(side="left", padx=(4, 12))
        self._level_box.bind("<<ComboboxSelected>>", self._on_level_selected)

        ttk.Checkbutton(toolbar, text="Auto-scroll", variable=self._autoscroll).pack(
            side="left"
        )
        ttk.Button(toolbar, text="Open Folder", command=self._open_folder).pack(
            side="right"
        )
        ttk.Button(toolbar, text="Clear", command=self.clear).pack(side="right", padx=4)

        body = ttk.Frame(self, padding=8)
        body.pack(fill="both", expand=True)

        # A plain tk.Text does not follow the ttkbootstrap theme, so its colours
        # are taken from the active theme.
        colors = ttk.Style().colors
        log_font = tkfont.nametofont("TkFixedFont").copy()
        log_font.configure(size=9)
        self._text = tk.Text(
            body,
            wrap="none",
            height=20,
            font=log_font,
            borderwidth=1,
            relief="solid",
            highlightthickness=0,
            background=colors.inputbg,
            foreground=colors.inputfg,
            insertbackground=colors.inputfg,
            selectbackground=colors.selectbg,
            selectforeground=colors.selectfg,
        )
        vertical = ttk.Scrollbar(body, orient="vertical", command=self._text.yview)
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=self._text.xview)
        self._text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

        self._text.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self._text.bind("<MouseWheel>", self._on_mouse_wheel)
        self._text.bind("<Button-4>", self._on_mouse_wheel)
        self._text.bind("<Button-5>", self._on_mouse_wheel)
        self._text.bind("<KeyPress>", self._block_editing)

    def start_polling(self, poll: Callable[[], list[str]]) -> None:
        """
        Pull log lines from *poll* until the window is closed.

        Parameters
        ----------
        poll : Callable[[], list[str]]
            Called on the main thread, returns the log lines to display.
        """
        self._poll = poll
        self._schedule_poll()

    def append_lines(self, lines: list[str]) -> None:
        """Append *lines* to the log area and drop the oldest ones if needed."""
        if not lines:
            return

        self._text.insert("end", "\n".join(lines) + "\n")
        self._trim_lines()
        if self._autoscroll.get():
            self._text.see("end")

    def clear(self) -> None:
        """Remove all lines from the log area."""
        self._text.delete("1.0", "end")

    def set_level(self, level: str) -> None:
        """Show *level* as the selected level without notifying the controller."""
        if level in LEVELS:
            self._level_box.set(level)

    def _schedule_poll(self) -> None:
        if self._poll is not None:
            self._after_id = self.after(POLL_INTERVAL_MS, self._poll_once)

    def _poll_once(self) -> None:
        self._after_id = None
        if self._poll is None:
            return

        try:
            self.append_lines(self._poll())
            self._schedule_poll()
        except tk.TclError:
            self._poll = None

    def _trim_lines(self) -> None:
        line_count = int(self._text.index("end-1c").split(".")[0])
        excess = line_count - MAX_LINES
        if excess > 0:
            self._text.delete("1.0", f"{excess + 1}.0")

    def _on_level_selected(self, _event: tk.Event | None = None) -> None:
        if self._on_level_change is not None:
            self._on_level_change(self._level_box.get())

    def _open_folder(self) -> None:
        if self._on_open_folder is not None:
            self._on_open_folder()

    def _on_mouse_wheel(self, event: tk.Event) -> str:
        if getattr(event, "num", None) == 4:
            self._text.yview_scroll(-1, "units")
            return "break"

        if getattr(event, "num", None) == 5:
            self._text.yview_scroll(1, "units")
            return "break"

        # Windows reports a notch as a multiple of 120, high resolution wheels
        # use smaller values that still mean a single notch. Tk only takes
        # whole numbers of units.
        if not event.delta:
            return "break"

        units = -int(event.delta / 120) if abs(event.delta) >= 120 else -1
        self._text.yview_scroll(units, "units")
        return "break"

    def _block_editing(self, event: tk.Event) -> str | None:
        """Prevent editing the log area while still allowing copy and select."""
        if event.state & CONTROL_MASK and (event.keysym or "").lower() in CONTROL_KEYS:
            return None

        if event.keysym in NAVIGATION_KEYS:
            return None

        return "break"

    def _on_close(self) -> None:
        self._poll = None
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        self.destroy()
