from typing import Protocol

from scored_lib.dart.dart_throw import DartThrow


class DartboardProtocol(Protocol):
    """Communication contract for dartboard-related UI consumers."""

    def addDartThrow(self, dart_throw: DartThrow) -> None:
        """Register a dart throw on the dartboard view."""

    def resetDartThrow(self) -> None:
        """Clear dartboard throw state and redraw the board."""
