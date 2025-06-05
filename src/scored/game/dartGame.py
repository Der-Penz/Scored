from dataclasses import dataclass
from typing import List, Tuple

from scored.board.dartboard import DartThrow

@dataclass(frozen=True)
class DartGameConfig:
    num_players: int = 2
    max_score: int = 501
    legs: int = 3
    sets: int = 1

class Leg: 
    def __init__(self, max_score: int = 0):
        self.score = max_score
        self.throws: List[DartThrow] = []

    def add_throw(self, throw: DartThrow) -> bool:
        """
        Adds a throw to the player's score. If the throw results in a score of 0,
        it indicates that the player has closed the leg.
        """
        self.throws.append(throw)
        result = self.score - throw.score

        if result == 0:
            self.score = result
            return True
        elif result > 0:
            self.score = result

        return False

    @property
    def get_round(self, round_index: int) -> List[DartThrow]:
        """
        Returns the throws for a specific round.
        """
        if round_index < 0 or round_index * 3 > len(self.throws):
            raise IndexError("Round index out of range.")
        return self.throws[round_index * 3 : (round_index + 1) * 3]
    
    @property
    def num_rounds(self) -> int:
        """
        Returns the number of rounds played by the player.
        """
        return len(self.throws) // 3

class Set:
    def __init__(self, max_score: int = 0, num_legs: int = 3):
        self.max_score = max_score
        self.legs: List[Leg] = [Leg(max_score) for _ in range(num_legs)]
        self.current_leg_index: int = 0

    @property
    def current_leg(self) -> Leg:
        """
        Returns the current leg being played.
        """
        return self.legs[self.current_leg_index]

    def next_leg(self):
        """
        Moves to the next leg in the set. If there are no more legs left, it raises an exception.
        """
        if self.current_leg_index < len(self.legs) - 1:
            self.current_leg_index += 1
        else:
            raise Exception("No more legs left in the set.")

class Player:
    def __init__(self, name: str, set: Set):
        self.name = name
        self.set = set

class DartGame:
    def __init__(self, config: DartGameConfig = DartGameConfig()):
        self.config = config
        self.players: List[Player] = [
            Player(f"Player {i + 1}", Set(config.max_score, config.legs))
            for i in range(config.num_players)
        ]
        self._current_player_index: int = 0
        self.throws_this_round = 0
        self.game_over = False

    @property
    def current_player(self) -> Player:
        return self.players[self._current_player_index]

    def next_player(self) -> Player:
        if self.game_over:
            raise Exception("Game is already over.")
        self._current_player_index = (self._current_player_index + 1) % self.config.num_players
        self.throws_this_round = 0
        return self.current_player

    def add_round(self, throws: List[DartThrow]):
        if self.game_over:
            raise Exception("Game is already over.")

        if len(throws) != 3:
            raise ValueError("A round must consist of exactly 3 darts.")
        
        leg = self.current_player.set.current_leg
        for throw in throws:
            if leg.add_throw(throw):
                self.next_leg()
                break

    def next_leg(self):
        for player in self.players:
            try:
                player.set.next_leg()
            except Exception as e:
                self.game_over = True
