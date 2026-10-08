from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from dataclasses_json import DataClassJsonMixin, dataclass_json


@dataclass_json
@dataclass
class LegAnnotation(DataClassJsonMixin):
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
        Initial score for the leg.
    is_won : bool | None, optional
        Status indicating whether the leg was won by this player, by default None.
    time_start : float | None, optional
        POSIX timestamp when the leg started, by default None.
    time_end : float | None, optional
        POSIX timestamp when the leg finished, by default None.
    """

    leg_id: str
    player_name: str
    starting_score: int
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
