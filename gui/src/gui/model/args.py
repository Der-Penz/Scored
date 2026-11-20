from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class AppMode(StrEnum):
    """The task the application is started for."""

    GAME = "game"
    LABEL = "label"

    @property
    def label(self) -> str:
        """Human readable name used in menus."""
        return "Play Game" if self is AppMode.GAME else "Label Images"


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""

    mode: str = field(
        default=AppMode.GAME.value,
        metadata={
            "choices": [mode.value for mode in AppMode],
            "help": "Run the game mode with the scoreboard and capture controls, "
            "or the label mode that steps through images to label their darts.",
        },
    )
    source: str | None = field(
        default=None,
        metadata={
            "help": "Specify the initial video source: a number for webcam, "
            "a path to a video file, or a URL for an HTTP stream."
        },
    )
    data_dir: Path | None = field(
        default=None,
        metadata={
            "help": "Enable data collection and store annotated data in this directory "
            "during game sessions. Leave unset to start with data collection disabled."
        },
    )
    label_dir: Path | None = field(
        default=None,
        metadata={
            "help": "Folder of images to open in label mode. Labels already stored "
            "next to an image are loaded, the others are labeled from scratch."
        },
    )
    read_only: bool = field(
        default=False,
        metadata={"help": "Open the images in label mode without writing any labels."},
    )
    log_dir: Path | None = field(
        default=None, metadata={"help": "Specify the directory to store log files."}
    )
    log_filename: str = field(
        default="scored_gui.log", metadata={"help": "The name of the log file."}
    )
    log_level: str = field(
        default="INFO",
        metadata={
            "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "help": "Set the logging level for the application.",
        },
    )

    @property
    def app_mode(self) -> AppMode:
        """The configured mode as an enum."""
        return AppMode(self.mode)
