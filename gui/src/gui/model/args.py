from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""

    source: str | None = None
    data_dir: Path | None = None
    log_dir: Path | None = None
    log_filename: str = "gui_scored.log"
    log_level: str = "INFO"
