from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

from scored_lib.dart.dart_throw import DartThrow, PositionSource
from scored_lib.dart.multiplier import Multiplier

ANNOTATION_FILENAME = "annotation.json"


@dataclass
class DartThrowAnnotation:
    """Annotation representation for a single dart throw.

    One annotation is stored per sample inside a leg directory, in an
    ``annotation.json`` file that sits next to the camera frame the annotation
    was made on.

    Parameters
    ----------
    throw_data : DartThrow
        The underlying dart throw object containing segment number, multiplier, and position.
    round : int
        One-based index of the round (visit) within the leg.
    throw : int
        One-based index of the throw within the round (1, 2, or 3).
    timestamp : float
        POSIX timestamp recording when the throw occurred.
    source : PositionSource
        Indicates whether the coordinates were derived manually or automatically.
    is_bust : bool
        Flag indicating if this throw resulted in a bust, by default False.
    leg_id: str
        Unique identifier for the leg session this throw belongs to.
    """

    throw_data: DartThrow
    round: int
    throw: int
    leg_id: str
    timestamp: float = field(default_factory=lambda: time())
    is_bust: bool = False

    @property
    def index(self) -> tuple[int, int]:
        """The one-based ``(round, throw)`` coordinates of this annotation.

        Returns
        -------
        tuple[int, int]
            The round index and the throw index within that round.
        """
        return (self.round, self.throw)

    def to_dict(self) -> dict[str, Any]:
        """Convert the annotation into a JSON serializable dictionary.

        Returns
        -------
        dict[str, Any]
            The annotation as plain Python types, enums replaced by their values.
        """
        position = self.throw_data.position
        return {
            "throw_data": {
                "number": self.throw_data.number,
                "multiplier": self.throw_data.multiplier.value,
                "position": tuple(position) if position is not None else None,
                "source": self.throw_data.source.value,
            },
            "round": self.round,
            "throw": self.throw,
            "timestamp": self.timestamp,
            "is_bust": self.is_bust,
            "leg_id": self.leg_id,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> DartThrowAnnotation:
        """Rebuild an annotation from a dictionary.

        Unknown keys are ignored so annotations stay readable when new fields
        are added later on.

        Parameters
        ----------
        raw : dict[str, Any]
            The dictionary as stored in ``annotation.json``.

        Returns
        -------
        DartThrowAnnotation
            The deserialized annotation.
        """
        throw_data = raw["throw_data"]
        position = throw_data["position"]

        return DartThrowAnnotation(
            round=raw["round"],
            throw=raw["throw"],
            timestamp=raw["timestamp"],
            is_bust=raw["is_bust"],
            leg_id=raw["leg_id"],
            throw_data=DartThrow(
                number=throw_data["number"],
                multiplier=Multiplier(throw_data["multiplier"]),
                position=tuple(position) if position is not None else None,
                source=PositionSource(throw_data["source"]),
            ),
        )

    def save(self, parent: Path):
        """Write the annotation to ``annotation.json``.

        Parameters
        ----------
        parent : Path
            The directory to write the annotation to.
        """
        parent.parent.mkdir(parents=True, exist_ok=True)

        with (parent / ANNOTATION_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @staticmethod
    def load(parent: Path) -> DartThrowAnnotation:
        """Read an annotation from an ``annotation.json`` file.

        Parameters
        ----------
        parent : Path
            The directory containing the annotation file.

        Returns
        -------
        DartThrowAnnotation
            The deserialized annotation.
        """
        with (parent / ANNOTATION_FILENAME).open("r", encoding="utf-8") as file:
            return DartThrowAnnotation.from_dict(json.load(file))
