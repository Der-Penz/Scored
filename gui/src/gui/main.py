import argparse
import sys
import tkinter as tk
from tkinter import ttk
import pywinstyles
import sv_ttk
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

def apply_theme_to_titlebar(root):
    # check if running on windows
    if sys.platform != "win32":
        return
    
    version = sys.getwindowsversion()

    if version.major == 10 and version.build >= 22000:
        pywinstyles.change_header_color(root, "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa")
    elif version.major == 10:
        pywinstyles.apply_style(root, "dark" if sv_ttk.get_theme() == "dark" else "normal")

        root.wm_attributes("-alpha", 0.99)
        root.wm_attributes("-alpha", 1)
        
    import ctypes

    try:
        # Forces sharp, native pixel rendering on Windows 10/11
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except AttributeError:
        # Fallback for older Windows environments
        ctypes.windll.user32.SetProcessDPIAware()

def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(AppConfig, dest="config")
    args = parser.parse_args()

    model = AppModel()
    root = tk.Tk()
    
    sv_ttk.set_theme(darkdetect.theme())
    apply_theme_to_titlebar(root)
    
    view = AppView(root)
    
    controller = AppController(view, model, args.config)
    controller.start()


if __name__ == "__main__":
    main()
