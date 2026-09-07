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
    round: int
    throw: int


@dataclass(slots=True)
class DartLeg:
    """Represent one player's leg as a sequence of individual throws."""

    starting_score: int = 501
    start_rule: StartRule = StartRule.ANY
    finish_rule: FinishRule = FinishRule.DOUBLE
    _results: list[list[ThrowResult | None]] = field(
        init=False, repr=False, default_factory=lambda: [[]]
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
    def current_turn_throws(self) -> tuple[ThrowResult, ...]:
        """
        Get the registered throw results of the current visible turn.

        Returns
        -------
        tuple of ThrowResult
            The non-None throw results of the last non-empty turn.
        """
        for turn in reversed(self._results):
            if turn:
                return tuple(tr for tr in turn if tr is not None)
        return ()

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
    
    @property
    def throw(self) -> int:
        """
        Get the current throw number within the round (1-based).

        Returns
        -------
        int
            The current throw number within the round.
        """
        if not self._results:
            return 1
        return len(self._results[-1]) + 1

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
            if len(self._results) >= 2:
                last_throws = self._results[-2]

        if len(last_throws) == 0:
            return self.starting_score

        for throw_result in last_throws[::-1]:  # Iterate in reverse to find the last non-None throw
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
        last_turn = self._results[-1]
        return 3 - len(last_turn)

    @property
    def rounds(self) -> tuple[tuple[ThrowResult, ...], ...]:
        """
        Get all throws grouped into rounds.

        Returns
        -------
        tuple of tuple of ThrowResult
            A tuple containing rounds, where each round is a tuple of at most three throws.
        """
        return tuple(tuple(round) for round in self._results)

    @property
    def is_finished(self) -> bool:
        return self._finished != (-1, -1)

    @property
    def finished_index(self) -> tuple[int, int]:
        """One-based (round, throw) index where the leg was finished, or (-1, -1)."""
        return self._finished

    @property
    def is_open(self) -> bool:
        """
        Check whether the leg is open for scoring.
        """
        return self._is_open

    def get_throw(self, round: int, throw: int) -> ThrowResult | None:
        """
        Get result of throw by a given round and throw

        Parameters
        ----------
        round : int
                one-based index of the round
        throw : int
                one-based index of the throw within the round

        Returns
        -------
        ThrowResult or None
                the result of the throw at the specified round and throw index, or None if the slot is a bust placeholder
        """
        if round < 1 or round > len(self._results):
            raise IndexError("round index out of range")
        if throw < 1 or throw > len(self._results[round - 1]):
            raise IndexError("throw index out of range")
        return self._results[round - 1][throw - 1]

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

        if self.is_open:
            if score_after < 0 or (
                score_after == 1
                and self.finish_rule
                in [FinishRule.DOUBLE, FinishRule.DOUBLE_OR_BULL, FinishRule.TRIPLE]
            ):
                bust = True
            elif score_after == 0:
                if self._matches_finish_rule(dart_throw):
                    finished = True
                    self._finished = (
                        len(self._results),
                        len(self._results[-1]) + 1 if self._results else 1,
                    )
                else:
                    bust = True

        if bust:
            first_throw_in_turn = True if self.throws_left == 3 else False
            if first_throw_in_turn:
                score_after = score_before
            else:
                score_after = self._results[-1][0].score_before

        throw_result = ThrowResult(
            dart_throw=dart_throw,
            score_before=score_before,
            score_after=score_after,
            opened_leg=opened_leg,
            bust=bust,
            finished=finished,
            round=len(self._results),
            throw=len(self._results[-1]) + 1,
        )
        self._results[-1].append(throw_result)
        if bust:
            left = self.throws_left
            for _ in range(left):
                self._results[-1].append(None)

        if next_round := self.throws_left == 0 and not finished:
            self._results.append([])

        next_round |= bust
        return throw_result, next_round

    def remove_last_throw(self) -> DartThrow | None:
        """
        Remove the last registered throw and restore the leg as if it was never thrown.

        The state is rebuilt by replaying all remaining throws, which also clears
        any bust placeholders, the finished flag and the leg-open state derived
        from the removed throw.

        Returns
        -------
        DartThrow | None
            The removed throw, or None if the leg has no throws to remove.
        """
        throws = [tr.dart_throw for turn in self._results for tr in turn if tr is not None]
        if not throws:
            return None

        removed = throws.pop()

        rebuilt = DartLeg(
            starting_score=self.starting_score,
            start_rule=self.start_rule,
            finish_rule=self.finish_rule,
        )
        for throw in throws:
            rebuilt.add_throw(throw)

        self._results = rebuilt._results
        self._is_open = rebuilt._is_open
        self._finished = rebuilt._finished

        return removed

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
        """Generate a structured visual log of the game rounds and score state."""
        lines = [
            f"DartLeg(Starting: {self.starting_score}, Rules: In-{self.start_rule.name}/Out-{self.finish_rule.name})"
        ]
        for i, round in enumerate(self._results):
            throw_strings = []
            final_turn_score = self.starting_score

            score = 0
            for tr in round:
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
