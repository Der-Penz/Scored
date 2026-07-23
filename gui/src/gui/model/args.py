from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""

    source: str | None = None # Specify the initial source: a number for webcam, a path to a video file, or a URL for an HTTP stream.
    # source: str | None = field(
    #     default=None,
    #     metadata={"help": "Specify the initial source: a number for webcam, a path to a video file, or a URL for an HTTP stream."},
    # )