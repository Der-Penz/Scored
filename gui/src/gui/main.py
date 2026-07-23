import argparse
import tkinter as tk

import multiprocessing as mp
import tkinter as tk

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

    # vars(args) converts the Namespace to a dictionary, which we unpack into the dataclass
    return AppConfig(**vars(args))


def main() -> None:
    mp.set_start_method("spawn")
    args = get_args()

    model = AppModel()
    root = tk.Tk()
    view = AppView(root)

    controller = AppController(model, view, args)
    controller.start()


if __name__ == "__main__":
    main()
