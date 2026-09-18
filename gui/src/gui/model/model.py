from gui.model.args import AppConfig
from scored_lib.game.game_leg import GameLeg
from scored_lib.game.player import Player


class AppModel:
    """Single source of truth for shared application state"""

    def __init__(self, args: AppConfig) -> None:
        self.args = args
        self.players: list[Player] = []
        self.game: GameLeg | None = None
