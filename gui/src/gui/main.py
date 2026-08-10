import argparse
import sys
import ttkbootstrap as ttk
import darkdetect
from simple_parsing import ArgumentParser

from gui.controller.app_controller import AppController
from gui.model.args import AppConfig
from gui.model.model import AppModel
from gui.view.app_view import AppView


def get_args() -> AppConfig:
    """
    Parses command line arguments for the application.

    Returns
    -------
    AppConfig
        The parsed command line arguments populated into a dataclass.
    """
    parser = argparse.ArgumentParser(description="Run the Scored GUI application.")

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        help="Specify the initial video source: a number for webcam, a path to a video file, or a URL for an HTTP stream.",
    )

    args = parser.parse_args()

    return AppConfig(**vars(args))


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(AppConfig, dest="config")
    args = parser.parse_args()

    model = AppModel()
    root = ttk.Window(
        title="Scored GUI",
        minsize=(600, 500),
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
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            if not hwnd:
                hwnd = root.winfo_id()

            # DWMWA_USE_IMMERSIVE_DARK_MODE attribute code
            # 20 is standard for Windows 11 / recent Win 10 builds
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
            )
        except Exception:
            pass

    view = AppView(root)

    controller = AppController(view, model, args.config)
    controller.start()


if __name__ == "__main__":
    main()
