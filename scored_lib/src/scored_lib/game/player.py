from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Player:
    """
    Represents a player participating in one or more dart matches.
    """

    name: str
    id: str = field(default_factory=lambda: str(uuid4()), compare=True)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Player name cannot be empty.")

    def __str__(self) -> str:
        return self.name
