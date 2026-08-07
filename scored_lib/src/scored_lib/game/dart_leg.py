from __future__ import annotations

from dataclasses import dataclass, field

from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier
from scored_lib.game.rule import FinishRule, StartRule
from scored_lib.util.finishes import get_best_checkout_path as lookup_best_checkout_path


@dataclass(frozen=True, slots=True)
class ThrowResult:
    """Computed outcome for a recorded dart throw."""

    dart_throw: DartThrow
    score_before: int
    score_after: int
    opened_leg: bool
    bust: bool
    finished: bool


@dataclass(slots=True)
class DartLeg:
    """Represent one player's leg as a sequence of individual throws."""

    starting_score: int = 501
    start_rule: StartRule = StartRule.ANY
    finish_rule: FinishRule = FinishRule.DOUBLE
    _results: list[list[ThrowResult | None]] = field(
        init=False, repr=False, default_factory=list
    )
    _is_open: bool = field(init=False, repr=False, default=False)
    _finished: tuple[int, int] = field(init=False, repr=False, default=(-1, -1))

    def __post_init__(self) -> None:
        """
        Validate initial parameters for the dart leg.

        Raises
        ------
        ValueError
            If starting_score is less than or equal to zero.
        """
        if self.starting_score <= 0:
            raise ValueError("starting_score must be greater than zero")

    @property
    def num_darts_thrown(self) -> int:
        """
        Get the total number of darts thrown in the leg.

        Returns
        -------
        int
            The total number of darts thrown.
        """
        return sum(len(turn) for turn in self._results)

    @property
    def round(self) -> int:
        """
        Get the current round number (1-based).

        Returns
        -------
        int
            The current round number.
        """
        return len(self._results) + 1

    def avg(self) -> float:
        """
        Calculate the average score per 3 darts thrown.

        Returns
        -------
        float
            The average score per 3 darts thrown. Returns 0.0 if no darts have been thrown.
        """
        total_throws = self.num_darts_thrown
        if total_throws == 0:
            return 0.0
        total_score = self.starting_score - self.score
        return (total_score / total_throws) * 3

    @property
    def score(self) -> int:
        """
        Get the current score of the player.
        """
        if len(self._results) == 0:
            return self.starting_score

        last_throws = self._results[-1]
        if len(last_throws) == 0:
            if len(self._results) == 1:
                last_throws = self._results[-2]

        if len(last_throws) == 0:
            return self.starting_score

        for throw_result in last_throws[
            ::-1
        ]:  # Iterate in reverse to find the last non-None throw
            if throw_result is not None:
                return throw_result.score_after

    @property
    def throws_left(self) -> int:
        """
        Get the number of throws left in the current turn.

        Returns
        -------
        int
            The number of throws left in the current turn (1 to 3).
        """
        if len(self._results) == 0:
            return 3
        last_turn = self._results[-1]
        if len(last_turn) == 3:
            return 3
        return 3 - len(last_turn)

    @property
    def turns(self) -> tuple[tuple[DartThrow, ...], ...]:
        """
        Get all throws grouped into turns.

        Returns
        -------
        tuple of tuple of DartThrow
            A tuple containing turns, where each turn is a tuple of at most three throws.
        """
        return tuple(tuple(turn) for turn in self._results)

    @property
    def is_finished(self) -> bool:
        return self._finished != (-1, -1)

    @property
    def finished_index(self) -> tuple[int, int]:
        return self._finished

    @property
    def is_open(self) -> bool:
        """
        Check whether the leg is open for scoring.
        """
        return self._is_open

    def get_throw(self, turn: int, throw: int) -> ThrowResult | None:
        """
        Get result of throw by a given turn and throw

        Parameters
        ----------
        turn : int
                zero-based index of the turn
        throw : int
                zero-based index of the throw within the turn

        Returns
        -------
        ThrowResult or None
                the result of the throw at the specified turn and throw index, or None if a throw is after a bust
        """
        if turn < 0 or turn >= len(self._results):
            raise IndexError("turn index out of range")
        if throw < 0 or throw >= len(self._results[turn]):
            raise IndexError("throw index out of range")
        return self._results[turn][throw]

    def get_best_checkout_path(self) -> tuple[DartThrow, ...] | None:
        """
        Get the best checkout path for a given score.

        Returns
        -------
        tuple of DartThrow or None
            The sequence of throws required to check out, or None if unavailable.
        """
        return lookup_best_checkout_path(self.score, self.finish_rule)

    def add_throw(self, dart_throw: DartThrow) -> tuple[ThrowResult, bool]:
        """
        Add a dart throw to the leg and compute the result.

        Parameters
        ----------
        dart_throw : DartThrow
            The throw to add.

        Returns
        -------
        tuple[ThrowResult, bool]
            The computed result of the throw and a boolean indicating whether a new round should be started.
        """
        if self.is_finished:
            raise ValueError("Cannot add throw to a finished leg.")

        score_before = self.score
        score_after = score_before - dart_throw.score

        opened_leg = False
        bust = False
        finished = False

        if not self.is_open:
            if self._matches_start_rule(dart_throw):
                opened_leg = True
                self._is_open = True
            else:
                bust = True
                score_after = score_before  # No change in score if not opened

        if self.is_open:
            if score_after < 0 or (
                score_after == 1
                and self.finish_rule
                in [FinishRule.DOUBLE, FinishRule.DOUBLE_OR_BULL, FinishRule.TRIPLE]
            ):
                bust = True
                score_after = score_before
            elif score_after == 0:
                if self._matches_finish_rule(dart_throw):
                    finished = True
                    self._finished = (
                        len(self._results),
                        len(self._results[-1]) if self._results else 0,
                    )
                else:
                    bust = True
                    score_after = score_before  # No change in score on bust

        throw_result = ThrowResult(
            dart_throw=dart_throw,
            score_before=score_before,
            score_after=score_after,
            opened_leg=opened_leg,
            bust=bust,
            finished=finished,
        )

        if self.throws_left == 3:
            self._results.append([])

        self._results[-1].append(throw_result)

        if bust:
            left = self.throws_left
            for _ in range(left):
                self._results[-1].append(None)

        new_round = bust or self.throws_left == 3
        return throw_result, new_round

    def _matches_start_rule(self, dart_throw: DartThrow) -> bool:
        """
        Check if a dart throw satisfies the start rule.

        Parameters
        ----------
        dart_throw : DartThrow
            The throw to check.

        Returns
        -------
        bool
            True if it satisfies the start rule, False otherwise.
        """
        if self.start_rule is StartRule.ANY:
            return True
        if self.start_rule is StartRule.DOUBLE:
            return dart_throw.multiplier is Multiplier.DOUBLE
        if self.start_rule is StartRule.TRIPLE:
            return dart_throw.multiplier is Multiplier.TRIPLE
        return False

    def _matches_finish_rule(self, dart_throw: DartThrow) -> bool:
        """
        Check if a dart throw satisfies the finish rule.

        Parameters
        ----------
        dart_throw : DartThrow
            The throw to check.

        Returns
        -------
        bool
            True if it satisfies the finish rule, False otherwise.
        """
        if self.finish_rule is FinishRule.ANY:
            return True
        if self.finish_rule is FinishRule.SINGLE:
            return dart_throw.multiplier is Multiplier.SINGLE
        if self.finish_rule is FinishRule.DOUBLE:
            return (
                dart_throw.multiplier is Multiplier.DOUBLE
                or dart_throw.multiplier is Multiplier.INNER_BULL
            )
        if self.finish_rule is FinishRule.TRIPLE:
            return dart_throw.multiplier is Multiplier.TRIPLE
        if self.finish_rule is FinishRule.DOUBLE_OR_BULL:
            return (
                dart_throw.multiplier is Multiplier.DOUBLE
                or dart_throw.multiplier is Multiplier.INNER_BULL
                or dart_throw.multiplier is Multiplier.OUTER_BULL
            )
        return False

    def __repr__(self) -> str:
        """Generate a structured visual log of the game turns and score state."""
        lines = [
            f"DartLeg(Starting: {self.starting_score}, Rules: In-{self.start_rule.name}/Out-{self.finish_rule.name})"
        ]
        for i, turn in enumerate(self._results):
            throw_strings = []
            final_turn_score = self.starting_score

            score = 0
            for tr in turn:
                if tr is None:
                    throw_strings.append("-")
                else:
                    label = str(tr.dart_throw.short_label)

                    suffix = " (BUST)" if tr.bust else ""
                    throw_strings.append(f"{label}{suffix}")
                    final_turn_score = tr.score_after
                    score += tr.dart_throw.score
                    if tr.bust:
                        score = 0  # Reset score to 0 for busts in the display

            throws_line = ", ".join(throw_strings)
            lines.append(
                f"  Turn {i + 1}: [{throws_line}] = {score} -> Remaining Score: {final_turn_score}"
            )

        lines.append(f"  Current Score: {self.score}")
        lines.append(f"  Average 3 Dart: {self.avg():.2f}")

        if self.is_finished:
            lines.append("  Status: FINISHED")
        return "\n".join(lines)
