from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

LEG_ANNOTATION_FILENAME = "leg.json"


@dataclass
class LegAnnotation:
    """General metadata describing a recorded leg.

    The metadata describes the leg as a whole and therefore lives at the top
    level of the leg directory, next to the per-sample folders.

    Parameters
    ----------
    leg_id : str
        Unique identifier for the leg session.
    player_name : str
        Name of the player throwing in this leg.
    starting_score : int, optional
        Initial score for the leg, by default 501.
    is_won : bool | None, optional
        Status indicating whether the leg was won by this player, by default None.
    time_start : float | None, optional
        POSIX timestamp when the leg started, by default None.
    time_end : float | None, optional
        POSIX timestamp when the leg finished, by default None.
    """

    leg_id: str
    player_name: str
    starting_score: int = 501
    is_won: bool | None = field(default=None, init=False)
    time_start: float = field(default_factory=lambda: time(), init=False)
    time_end: float | None = field(default=None, init=False)

    @property
    def is_finished(self) -> bool:
        """Check if the leg has finished.

        Returns
        -------
        bool
            True if the leg has finished, False otherwise.
        """
        return self.time_end is not None

    def end(self, time_end: float, is_won: bool) -> None:
        """Mark the leg as finished.

        Parameters
        ----------
        time_end : float
            POSIX timestamp indicating when the leg finished.
        is_won : bool
            Whether the player won the leg.
        """
        self.time_end = time_end
        self.is_won = is_won

    def to_dict(self) -> dict[str, Any]:
        """Convert the leg annotation into a JSON serializable dictionary.

        Returns
        -------
        dict[str, Any]
            The leg annotation as plain Python types.
        """
        return {
            "leg_id": self.leg_id,
            "player_name": self.player_name,
            "starting_score": self.starting_score,
            "is_won": self.is_won,
            "time_start": self.time_start,
            "time_end": self.time_end,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> LegAnnotation:
        """Rebuild leg annotation from a dictionary.

        Unknown keys are ignored so leg files stay readable when new fields are
        added later on.

        Parameters
        ----------
        raw : dict[str, Any]
            The dictionary as stored in ``leg.json``.

        Returns
        -------
        LegInfo
            The deserialized leg annotation.
        """
        return LegAnnotation(
            leg_id=raw["leg_id", ""],
            player_name=raw["player_name", ""],
            starting_score=raw["starting_score", 501],
            is_won=raw["is_won"],
            time_start=raw["time_start"],
            time_end=raw["time_end"],
        )

    def save(self, dir: Path) -> Path:
        """Write the leg annotation to disk.

        Parameters
        ----------
        dir : Path
            The directory to write the leg annotation to.

        Returns
        -------
        Path
            The path that was written.
        """
        dir.parent.mkdir(parents=True, exist_ok=True)

        with (dir / LEG_ANNOTATION_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)
        return dir / LEG_ANNOTATION_FILENAME

    @staticmethod
    def load(path: Path) -> LegAnnotation:
        """Read leg annotation from disk.

        Parameters
        ----------
        path : Path
            The leg annotation file to read.

        Returns
        -------
        LegInfo
            The deserialized leg annotation.
        """
        with Path(path).open("r", encoding="utf-8") as file:
            return LegAnnotation.from_dict(json.load(file))
