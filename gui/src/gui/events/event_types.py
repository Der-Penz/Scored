from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scored_lib.dart.dart_throw import DartThrow


@dataclass(frozen=True)
class DartThrowEvent:
    """Notification that a new dart throw is available in AppModel.pending_throw."""
    throw : DartThrow


@dataclass(frozen=True)
class PlayerAddedEvent:
    """Notification that a new player was added (see AppModel.players)."""

@dataclass(frozen=True)
class PlayersChangedEvent:
    """Notification that the player list changed (see AppModel.players)."""


@dataclass(frozen=True)
class GameStartedEvent:
    """Notification that a new game was started (see AppModel.legs)."""

@dataclass(frozen=True)
class FrameCapturedEvent:
    """Carries the captured frame (too transient for model storage)."""

    frame: np.ndarray
