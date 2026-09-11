from typing import Protocol, Sequence

from scored_lib.game.dart_leg import ThrowResult


class DartboardProtocol(Protocol):
    """Communication contract for dartboard-related UI consumers."""

    def draw_darts(self, throws: Sequence[ThrowResult]) -> None:
        """Draw the given dart throws on the dartboard view."""

    def clear(self) -> None:
        """Clear dartboard throw state and redraw the board."""
