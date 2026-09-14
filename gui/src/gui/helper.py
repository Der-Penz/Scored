import tkinter as tk


def center_dialog(window: tk.Toplevel, parent: tk.Misc) -> None:
    """Center *window* over *parent* on screen."""
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
    window.geometry(f"+{x_pos}+{y_pos}")
