from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""

    source: str | None = None
    data_dir: str | None = None
