from dataclasses import dataclass, field
from typing import Sequence

from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier
from scored_lib.game.dart_leg import DartLeg, ThrowResult
from scored_lib.game.rule import FinishRule, StartRule


@dataclass(slots=True)
class DartGame:
    """Coordinate a leg for each player and advance turns across them."""

    starting_score: int = 501
    player_names: Sequence[str] = field(default_factory=lambda: ("Player 1",))
    start_rule: StartRule = StartRule.ANY
    finish_rule: FinishRule = FinishRule.DOUBLE
    _legs: list[DartLeg] = field(init=False, repr=False, default_factory=list)
    _current_player_index: int = field(init=False, repr=False, default=0)
    _current_turn_index: int = field(init=False, repr=False, default=0)
    _pending_turn: list[DartThrow] = field(init=False, repr=False, default_factory=list)
    _finished: bool = field(init=False, repr=False, default=False)
    _winner_index: int | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.starting_score <= 0:
            raise ValueError("starting_score must be greater than zero")

        names = tuple(self.player_names)
        if not names:
            raise ValueError("player_names must not be empty")
        if any(not name for name in names):
            raise ValueError("player names must not be empty")

        self.player_names = names
        self._legs = [
            DartLeg(
                player_name=name,
                starting_score=self.starting_score,
                start_rule=self.start_rule,
                finish_rule=self.finish_rule,
            )
            for name in names
        ]

    @property
    def legs(self) -> tuple[DartLeg, ...]:
        return tuple(self._legs)

    @property
    def current_player_index(self) -> int:
        return self._current_player_index

    @property
    def current_player_name(self) -> str:
        return self._legs[self._current_player_index].player_name

    @property
    def current_leg(self) -> DartLeg:
        return self._legs[self._current_player_index]

    @property
    def current_turn_index(self) -> int:
        return self._current_turn_index

    @property
    def current_dart_index(self) -> int:
        return len(self._pending_turn)

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def winner_index(self) -> int | None:
        return self._winner_index

    @property
    def winner_name(self) -> str | None:
        if self._winner_index is None:
            return None
        return self._legs[self._winner_index].player_name

    @property
    def current_score(self) -> int:
        if not self._pending_turn:
            return self.current_leg.score
        return self._preview_pending_turn(self._pending_turn)[0][-1].score_after

    @property
    def player_scores(self) -> tuple[int, ...]:
        scores = [leg.score for leg in self._legs]
        if self._pending_turn:
            scores[self._current_player_index] = self.current_score
        return tuple(scores)

    def get_leg(self, player_index: int) -> DartLeg:
        return self._legs[player_index]

    def get_best_checkout_path(
        self, player_index: int | None = None
    ) -> tuple[DartThrow, ...] | None:
        if player_index is None:
            return self.current_leg.get_best_checkout_path(self.current_score)
        return self.get_leg(player_index).get_best_checkout_path()

    def get_checkout_path(
        self, player_index: int | None = None
    ) -> tuple[DartThrow, ...] | None:
        return self.get_best_checkout_path(player_index)

    def record_throw(self, dart_throw: DartThrow) -> ThrowResult:
        if self._finished:
            raise RuntimeError("The game is already finished")

        self._pending_turn.append(dart_throw)
        results, finished, bust = self._preview_pending_turn(self._pending_turn)
        result = results[-1]

        if finished or bust or len(self._pending_turn) >= 3:
            self.current_leg.record_turn(self._pending_turn)
            self._pending_turn = []
            if finished:
                self._finished = True
                self._winner_index = self._current_player_index
            else:
                self._advance_turn()

        return result

    def _advance_turn(self) -> None:
        self._current_turn_index += 1
        self._current_player_index = (self._current_player_index + 1) % len(self._legs)

    def _preview_pending_turn(
        self, dart_throws: Sequence[DartThrow]
    ) -> tuple[tuple[ThrowResult, ...], bool, bool]:
        leg = self.current_leg
        turn_score = leg.score
        turn_start_score = turn_score
        leg_open = leg.is_open or leg.start_rule is StartRule.ANY
        results: list[ThrowResult] = []
        finished = False
        bust = False
        history_index = len(leg.throws)

        for dart_index_in_turn, dart_throw in enumerate(dart_throws, start=1):
            score_before = turn_score
            opened_leg = False
            applied_score = 0

            if leg_open:
                applied_score = dart_throw.score
            elif self._matches_start_rule(dart_throw):
                leg_open = True
                opened_leg = True
                applied_score = dart_throw.score

            score_after = score_before - applied_score

            if score_after < 0:
                bust = True
                score_after = turn_start_score
            elif score_after == 0:
                if self._matches_finish_rule(dart_throw):
                    finished = True
                else:
                    bust = True
                    score_after = turn_start_score

            results.append(
                ThrowResult(
                    history_index=history_index + dart_index_in_turn - 1,
                    dart_throw=dart_throw,
                    score_before=score_before,
                    applied_score=applied_score,
                    score_after=score_after,
                    opened_leg=opened_leg,
                    bust=bust,
                    finished=finished,
                    is_empty_throw=False,
                ),
            )
            turn_score = score_after

            if bust or finished:
                break

        return tuple(results), finished, bust

    def _matches_start_rule(self, dart_throw: DartThrow) -> bool:
        if self.start_rule is StartRule.ANY:
            return True
        if self.start_rule is StartRule.DOUBLE:
            return dart_throw.multiplier is Multiplier.DOUBLE
        if self.start_rule is StartRule.TRIPLE:
            return dart_throw.multiplier is Multiplier.TRIPLE
        return False

    def _matches_finish_rule(self, dart_throw: DartThrow) -> bool:
        if self.finish_rule is FinishRule.ANY:
            return True
        if self.finish_rule is FinishRule.SINGLE:
            return dart_throw.multiplier is Multiplier.SINGLE
        if self.finish_rule is FinishRule.DOUBLE:
            return dart_throw.multiplier is Multiplier.DOUBLE
        if self.finish_rule is FinishRule.TRIPLE:
            return dart_throw.multiplier is Multiplier.TRIPLE
        if self.finish_rule is FinishRule.DOUBLE_OR_BULL:
            return (
                dart_throw.multiplier is Multiplier.DOUBLE
                or dart_throw.multiplier is Multiplier.INNER_BULL
            )
        return False
