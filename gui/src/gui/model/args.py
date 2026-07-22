from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    source: str | None = None