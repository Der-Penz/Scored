from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""

    source: str | None = field(
        default=None,
        metadata={
            "help": "Specify the initial video source: a number for webcam, a path to a video file, or a URL for an HTTP stream."
        },
    )
    data_dir: Path | None = field(
        default=None,
        metadata={
            "help": "Enable data collection and store annotated data in this directory during game sessions. Leave unset to start with data collection disabled."
        },
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
