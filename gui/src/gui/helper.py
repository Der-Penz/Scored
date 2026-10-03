import os
import platform
import subprocess
import tkinter as tk
from pathlib import Path


def center_dialog(window: tk.Toplevel, parent: tk.Misc) -> None:
    """Center *window* over *parent* on screen."""
    # 1. Inform the Window Manager / Wayland compositor that this is a child dialog
    window.transient(parent)

    # 2. Set window type hint for XWayland/Linux WMs
    try:
        window.wm_attributes("-type", "dialog")
    except tk.TclError:
        pass  # Platform doesn't support -type attribute

    window.update_idletasks()
    window_w, window_h = window.winfo_width(), window.winfo_height()

    try:
        x = parent.winfo_rootx()
        y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
    except tk.TclError:
        x = y = 0
        parent_w = parent_h = 0

    x_pos = x + max(0, (parent_w - window_w) // 2)
    y_pos = y + max(0, (parent_h - window_h) // 2)

    # Geometry positioning (works via XWayland when transient is set)
    window.geometry(f"+{x_pos}+{y_pos}")


def open_in_file_manager(path: Path | str) -> None:
    """Open *path* with the file manager of the current system."""
    target = Path(path)
    system = platform.system()

    if system == "Windows":
        os.startfile(str(target))
        return

    program = "open" if system == "Darwin" else "xdg-open"
    subprocess.run(
        [program, str(target)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
