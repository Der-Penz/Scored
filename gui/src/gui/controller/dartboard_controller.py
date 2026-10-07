import logging
import tkinter as tk

from scored_lib.dart.constants import Position
from scored_lib.dart.dart_throw import PositionSource
from scored_lib.dart.scoring import score_dart_throw
from scored_lib.game.dart_leg import ThrowResult
from scored_lib.util.position import canvas_to_relative_position

from gui.events.event_channel import EventChannel
from gui.events.event_types import (
    DartThrowEvent,
    GameStateChanged,
    ThrowEdited,
    TurnChanged,
)
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView

DRAG_RELEASE_DEBOUNCE_MS = 150


class DartboardController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._darts: list[ThrowResult] = []
        self._debounce_after_id: str | None = None

    def bind_menu(self, menu: tk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        self._event_channel.subscribe(TurnChanged, lambda _: self.clear())
        self._event_channel.subscribe(GameStateChanged, self.on_game_state_changed)
        self.on_panel_rebound(Panel.DARTBOARD, self._restore_board)

    def clear(self) -> None:
        self._cancel_debounce()
        self._darts.clear()
        self._view.dartboard_view.clear()

    def on_game_state_changed(self, _: GameStateChanged) -> None:
        if self._model.game is None:
            self.clear()
            return

        self._darts = list(self._model.game.current_leg.current_turn_throws)
        self._redraw_darts()

    def _redraw_darts(self) -> None:
        self._view.dartboard_view.draw_darts(self._darts)

    def _on_board_resized(self) -> None:
        """Re-draw the current dart list at the new board geometry."""
        self._redraw_darts()

    def _on_drag_release(self, index: int, position: Position) -> None:
        """Handle a dart marker drag release, updating the model."""
        if index < 0 or index >= len(self._darts):
            return

        self._cancel_debounce()
        self._debounce_after_id = self._view.after(
            DRAG_RELEASE_DEBOUNCE_MS,
            lambda idx=index, pos=position: self._apply_dart_drag(idx, pos),
        )

    def _on_board_click(self, position: Position) -> None:
        if self._model.game is None:
            return

        scoring_position = canvas_to_relative_position(position)
        scored = score_dart_throw(scoring_position, PositionSource.MANUAL)

        self._event_channel.emit(DartThrowEvent(throw=scored))

    def _apply_dart_drag(self, index: int, new_position: Position) -> None:
        assert self._model.game is not None

        self._debounce_after_id = None

        scoring_position = canvas_to_relative_position(new_position)
        scored = score_dart_throw(scoring_position, PositionSource.MANUAL)

        previous = self._darts[index].dart_throw
        result = self._model.game.current_leg.edit_current_throw(
            scored, throw=index + 1
        )

        logging.info(f"Edited throw: {previous} -> {scored}")

        self._event_channel.emit(
            ThrowEdited(
                player=self._model.game.current_player,
                result=result,
                previous=previous,
            )
        )
        self._event_channel.emit(GameStateChanged())

    def _cancel_debounce(self) -> None:
        if self._debounce_after_id is not None:
            self._view.after_cancel(self._debounce_after_id)
            self._debounce_after_id = None

    def start(self) -> None:
        self._restore_board()

    def _bind_callbacks(self) -> None:
        """Point the live dartboard panel at this controller."""
        board = self._view.dartboard_view
        board.set_resize_callback(self._on_board_resized)
        board.set_drag_release_callback(self._on_drag_release)
        board.set_board_click_callback(self._on_board_click)

    def _restore_board(self) -> None:
        """Hook up and repaint the dartboard after it was rebuilt."""
        self._cancel_debounce()
        self._bind_callbacks()
        self._redraw_darts()
