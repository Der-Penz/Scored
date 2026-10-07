from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.game.dart_leg import ThrowResult
from scored_lib.game.player import Player

from gui.model.layout import Panel


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
class GameStateChanged:
    """
    Notification that the game state changed and views should redraw.
    """


@dataclass(frozen=True)
class ThrowRecorded:
    """Notification that a throw was registered for the given player."""

    player: Player
    result: ThrowResult


@dataclass(frozen=True)
class ThrowEdited:
    """Notification that an already registered throw was replaced."""

    player: Player
    result: ThrowResult
    previous: DartThrow


@dataclass(frozen=True)
class ThrowRemoved:
    """
    Notification that a registered throw was removed again.
    """

    player: Player
    dart_throw: DartThrow
    round: int
    throw: int


@dataclass(frozen=True)
class TurnChanged:
    """Notification that the current player changed (see AppModel.game)."""


@dataclass(frozen=True)
class UndoRequested:
    """Notification that the user wants to undo the last registered throw."""


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


@dataclass(frozen=True)
class PanelRebound:
    """
    Notification that a panel's widgets were built again for a new host.

    Tk cannot move an existing widget to another parent, so a panel that changes
    column or is floated has to be recreated. Controllers holding a reference to
    the old widget use this to re-acquire it and repaint from the model.
    """

    panel: Panel


@dataclass(frozen=True)
class PanelClosed:
    """Notification that the user closed the window of a floating panel."""

    panel: Panel
