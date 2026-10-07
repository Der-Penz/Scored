from __future__ import annotations

import logging
import queue
import sys
import tkinter as tk
from collections import deque

import ttkbootstrap as ttk
from scored_lib.logging_setup import (
    create_formatter,
    get_log_file,
    resolve_caller,
    set_log_level,
)

from gui.events.event_channel import EventChannel
from gui.helper import open_in_file_manager
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from gui.view.misc.logging_view import LoggingView

QUEUE_SIZE = 5000
PRELOAD_LINES = 500


class QueueLogHandler(logging.Handler):
    """Log handler that hands the formatted records over to the GUI thread."""

    def __init__(self, records: queue.Queue[str]) -> None:
        super().__init__()
        # The location is resolved in emit(), the caller stack is gone once the
        # records reach the view.
        self.setFormatter(create_formatter(resolve_location=False))
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        """Format *record* and queue it, dropping it when the queue is full."""
        try:
            record.filename, record.lineno = resolve_caller(sys._getframe(1))
            self._records.put_nowait(self.format(record))
        except queue.Full:
            pass
        except Exception:
            self.handleError(record)


class LoggingController(BaseController):
    """
    Shows the application log in a scrollable window and opens the log folder.

    Records are formatted on the emitting thread and buffered in a queue, so
    this controller never touches Tk from a background thread; the view pulls
    the buffered lines on the main thread.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        set_log_level(model.config.log_level)

        self._records: queue.Queue[str] = queue.Queue(maxsize=QUEUE_SIZE)
        self._handler = QueueLogHandler(self._records)
        self._log_window: LoggingView | None = None

    def bind_menu(self, menu: ttk.Menu) -> None:
        menu.add_command(label="Logs...", command=self._open_logs, accelerator="Ctrl+L")
        menu.add_separator()
        menu.add_command(label="Open Log Folder", command=self._open_log_folder)

    def bind_components(self) -> None:
        logging.getLogger().addHandler(self._handler)
        self._view.master.bind_all("<Control-l>", lambda _: self._open_logs())

    def start(self) -> None:
        logging.debug("Log window is available under Logs... (Ctrl+L)")

    def _open_logs(self) -> None:
        """Open the log window, or bring it to the front if it is already open."""
        window = self._log_window
        if window is not None and self._window_open(window):
            window.deiconify()
            window.lift()
            return

        self._log_window = LoggingView(
            self._view,
            level=logging.getLevelName(logging.getLogger().level),
            on_level_change=self.set_level,
            on_open_folder=self._open_log_folder,
        )
        self._log_window.append_lines(self._preload_lines())
        self._log_window.start_polling(self._collect_records)

    def _open_log_folder(self) -> None:
        """Show the log file in the system file manager."""
        log_file = get_log_file()
        if log_file is None:
            ttk.Messagebox.show_info(
                message="File logging is disabled, there is no log folder.",
                title="Logs",
                parent=self._view,
            )
            return

        open_in_file_manager(log_file.parent)

    def set_level(self, level: str) -> None:
        """Change the log level of the running application."""
        applied = set_log_level(level)
        logging.info(f"Log level set to {logging.getLevelName(applied)}")

    def _window_open(self, window: LoggingView) -> bool:
        """Check whether *window* still exists, it may have been closed."""
        try:
            return bool(window.winfo_exists())
        except tk.TclError:
            return False

    def _collect_records(self) -> list[str]:
        """Drain the buffered log lines, called by the view on the main thread."""
        lines: list[str] = []
        while True:
            try:
                lines.append(self._records.get_nowait())
            except queue.Empty:
                return lines

    def _preload_lines(self) -> list[str]:
        """Read the tail of the current log file so the window is never empty."""
        log_file = get_log_file()
        if log_file is None or not log_file.exists():
            return []

        with log_file.open("r", encoding="utf-8", errors="replace") as file:
            lines = deque(file, maxlen=PRELOAD_LINES)

        return [line.rstrip("\n") for line in lines]
