from scored_lib.game.dart_leg import DartLeg
from scored_lib.game.player import Player


class AppModel:
    """Single source of truth for shared application state"""

    def __init__(self) -> None:
        self.players: list[Player] = []
        self.legs: dict[str, DartLeg] = {}
        self.current_player_index: int = 0
