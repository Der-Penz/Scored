from __future__ import annotations

from dataclasses import dataclass, field

from scored_lib.game.dart_leg import DartLeg, ThrowResult
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.game.player import Player
from scored_lib.game.rule import FinishRule, StartRule


@dataclass(slots=True)
class GameLeg:
    """
    Represents one leg between multiple players.

    Each player owns an independent ``DartLeg`` that tracks their
    score and throws. This class only manages turn order and determines
    the winner.
    """

    players: tuple[Player, ...]

    starting_score: int = 501
    start_rule: StartRule = StartRule.ANY
    finish_rule: FinishRule = FinishRule.DOUBLE

    starting_player: int = 0

    _legs: dict[Player, DartLeg] = field(init=False, repr=False)
    _current_player: int = field(init=False, repr=False)
    _winner: Player | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if len(self.players) < 1:
            raise ValueError("At least one player is required.")

        if not (0 <= self.starting_player < len(self.players)):
            raise ValueError("Invalid starting player.")

        self._current_player = self.starting_player

        self._legs = {
            player: DartLeg(
                starting_score=self.starting_score,
                start_rule=self.start_rule,
                finish_rule=self.finish_rule,
            )
            for player in self.players
        }

    @property
    def current_player(self) -> Player:
        return self.players[self._current_player]

    @property
    def winner(self) -> Player | None:
        return self._winner

    @property
    def is_finished(self) -> bool:
        return self._winner is not None

    @property
    def current_leg(self) -> DartLeg:
        return self._legs[self.current_player]

    def leg_for(self, player: Player) -> DartLeg:
        return self._legs[player]

    def add_throw(self, dart_throw: DartThrow) -> tuple[Player, ThrowResult, bool]:
        """
        Record a throw for the current player.

        Does not advance to the next player after the turn ends; call
        :meth:`next_player` once the finished turn is confirmed.

        Parameters
        ----------
        dart_throw : DartThrow
            The throw to record for the current player.

        Returns
        -------
        tuple[Player, ThrowResult, bool]
            A tuple containing the player who made the throw, the result of the throw,
            and a boolean indicating whether the turn has ended.
        """

        if self.is_finished:
            raise ValueError("This leg has already finished.")

        player = self.current_player
        leg = self._legs[self.current_player]

        result, end_turn = leg.add_throw(dart_throw)

        if result.finished:
            self._winner = self.current_player
            return player, result, True

        return player, result, end_turn

    def next_player(self) -> Player:
        """
        Advance to the next player.

        Call this to confirm a completed turn before it moves to the next player.

        Returns
        -------
        Player
            The player whose turn is active afterwards.
        """
        self._current_player = (self._current_player + 1) % len(self.players)
        return self.current_player

    def undo_last_throw(self) -> DartThrow | None:
        """
        Remove the last registered throw of the current player's leg.

        Returns
        -------
        DartThrow | None
            The removed throw, or None if there was nothing to remove.
        """
        return self.current_leg.remove_last_throw()

    def standings(self) -> list[tuple[Player, int]]:
        """
        Returns players ordered by remaining score.
        """
        return sorted(
            ((player, leg.score) for player, leg in self._legs.items()),
            key=lambda x: x[1],
        )

    def __repr__(self) -> str:
        lines = ["GameLeg"]

        for i, player in enumerate(self.players):
            leg = self._legs[player]

            markers = []
            if i == self._current_player and not self.is_finished:
                markers.append(f"TURN ({leg.throws_left})")
            if self._winner is player:
                markers.append("WINNER")

            suffix = f" ({', '.join(markers)})" if markers else ""

            lines.append(
                f"  {player.name:<12} Score: {leg.score:>3}  Avg: {leg.avg():>6.2f}{suffix}"
            )

        return "\n".join(lines)
