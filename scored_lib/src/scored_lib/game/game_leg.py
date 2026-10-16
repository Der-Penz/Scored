from __future__ import annotations

import logging
from dataclasses import dataclass, field

from scored_lib.dart.dart_throw import DartThrow
from scored_lib.game.dart_leg import DartLeg, ThrowResult
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
    def last_player(self) -> Player:
        """
        Returns the player who threw most recently.

        If no throw has been registered yet, returns the starting player.
        """
        owner = self._owner_of_last_throw()
        if owner is None:
            return self.players[self.starting_player]
        return self.players[owner]

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
            logging.info(f"{player.name} won the game")
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
        leg = self.leg_for(self.current_player)
        if leg.throws_left == 0:
            leg.next_turn()
        return self.current_player

    def undo_last_throw(self) -> ThrowResult | None:
        """
        Remove the most recently registered throw, wherever it was thrown.

        The active player is restored to whoever made the removed throw. If
        that throw had finished the game, the winner is cleared again. If there
        is nothing to remove, the active player is left untouched.

        Returns
        -------
        ThrowResult | None
            The result the removed throw had, including the ``(round, throw)``
            coordinates it occupied, or None if there was nothing to remove.
            The coordinates refer to the state before the removal.
        """
        if self.winner is not None:
            raise ValueError("Cannot undo a throw after the leg has finished.")

        owner = self._owner_of_last_throw()
        if owner is None:
            logging.debug("Nothing to undo, no throw has been registered yet")
            return None

        self._current_player = owner
        removed = self.current_leg.remove_last_throw()

        if removed is not None:
            logging.info(f"Removed last throw {removed.dart_throw.short_label}")

        return removed

    def _owner_of_last_throw(self) -> int | None:
        """
        Find the player whose throw was registered most recently.

        Returns
        -------
        int | None
            The index of that player, or None if no throw has been registered
            in this game at all.
        """
        if self.current_leg.throws_left != 3:
            # The active player threw within their current turn, so their most
            # recent throw is also the most recent one of the whole game.
            return self._current_player

        # The active player's current turn is still empty, so the most recent
        # throw belongs to a player who already had their turn.
        for offset in range(1, len(self.players)):
            index = (self._current_player - offset) % len(self.players)
            if self._legs[self.players[index]].num_darts_thrown:
                return index

        # Nobody else has a throw; in a one player game the active player can
        # still have one from a previous turn.
        if self.current_leg.num_darts_thrown:
            return self._current_player

        return None

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
