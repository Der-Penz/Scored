import numpy as np
from scored_lib.annotation.image_annotation import ImageAnnotation
from scored_lib.game.game_leg import GameLeg
from scored_lib.game.player import Player

from gui.model.args import AppConfig


class AppModel:
    """Single source of truth for shared application state"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.players: list[Player] = []
        self.game: GameLeg | None = None
        self.annotations: list[tuple[ImageAnnotation, np.ndarray]] = []
