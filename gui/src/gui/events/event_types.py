from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.game.player import Player


@dataclass(frozen=True)
class DartThrowEvent:
    """Carries a new dart throw that should be registered (see AppModel.game)."""

    throw: DartThrow


@dataclass(frozen=True)
class PlayerAdded:
    """Carries the player that was added to AppModel.players."""

    player: Player


@dataclass(frozen=True)
class PlayerRemoved:
    """Carries the player that was removed from AppModel.players."""

    player: Player


@dataclass(frozen=True)
class ScoreChanged:
    """Notification that scores changed (see AppModel.game)."""


@dataclass(frozen=True)
class ThrowEdited:
    """Notification that a throw was edited (see AppModel.game)."""

    throw: int


@dataclass(frozen=True)
class TurnChanged:
    """Notification that the current player changed (see AppModel.game)."""


@dataclass(frozen=True)
class GameStarted:
    """Notification that a new game was started (see AppModel.game)."""


@dataclass(frozen=True)
class GameEnded:
    """Notification that the current game ended (see AppModel.game)."""


@dataclass(frozen=True)
class FrameCapturedEvent:
    """Carries the captured frame (too transient for model storage)."""

    frame: np.ndarray
