import argparse
import logging
import sys
from pathlib import Path

import darkdetect
import ttkbootstrap as ttk
from scored_lib.logging_setup import setup_logging
from simple_parsing import ArgumentParser

from gui.controller.app_controller import AppController
from gui.model.args import AppConfig
from gui.model.model import AppModel
from gui.view.app_view import AppView


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(AppConfig, dest="config")
    args = parser.parse_args()

    log_file = setup_logging(
        log_dir=args.config.log_dir,
        filename=args.config.log_filename,
        level=args.config.log_level,
    )

    logging.info("Starting Scored GUI")
    logging.info(f"Config: {args.config}")
    logging.info(f"Logging to file: {log_file if log_file else 'disabled'}")

    model = AppModel(config=args.config)
    root = ttk.Window(
        title="Scored GUI",
        minsize=None,
        theme=f"bootstrap-{'dark' if darkdetect.isDark() else 'light'}",
    )
    is_dark = darkdetect.isDark()
    # Force dark title bar frame directly on the window handle
    if sys.platform.startswith("win") and is_dark:
        import ctypes

        try:
            # Force Tkinter to fully initialize the window frame manager first
            root.update_idletasks()

            # With ttkbootstrap, the root winfo_id maps directly to the target window handle
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())  # type: ignore
            if not hwnd:
                hwnd = root.winfo_id()

            # DWMWA_USE_IMMERSIVE_DARK_MODE attribute code
            # 20 is standard for Windows 11 / recent Win 10 builds
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(  # type: ignore
                hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
            )
        except Exception:
            pass

    view = AppView(root)

    controller = AppController(view, model, args.config)
    controller.start()

    logging.info("Scored GUI stopped")


if __name__ == "__main__":
    main()
