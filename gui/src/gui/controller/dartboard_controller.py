import tkinter as tk

from gui.events.event_channel import EventChannel
from gui.events.event_types import ScoreChanged, TurnChanged
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from scored_lib.dart.constants import Position
from scored_lib.dart.scoring import score_dart_throw
from scored_lib.game.dart_leg import ThrowResult
from scored_lib.util.position import canvas_to_relative_position

DRAG_RELEASE_DEBOUNCE_MS = 150


class DartboardController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._left_view = view.left_view
        self._dartboard_view = view.left_view.dartboard_view
        self._darts: list[ThrowResult] = []
        self._debounce_after_id: str | None = None

    def bind_menu(self, menu: tk.Menu) -> None:
        self._dartboard_visible = tk.BooleanVar(value=True)
        menu.add_checkbutton(
            label="Show Dartboard",
            variable=self._dartboard_visible,
            command=self._toggle_dartboard_visibility,
        )

    def bind_components(self) -> None:
        self._event_channel.subscribe(TurnChanged, lambda _: self.clear())
        self._event_channel.subscribe(ScoreChanged, self.on_score_changed)

    def clear(self) -> None:
        self._cancel_debounce()
        self._darts.clear()
        self._dartboard_view.clear()

    def on_score_changed(self, _) -> None:
        if self._model.game is None:
            self.clear()
            return

        self._darts = list(self._model.game.current_leg.current_turn_throws)
        self._redraw_darts()

    def _redraw_darts(self) -> None:
        self._dartboard_view.draw_darts(self._darts)

    def _on_board_resized(self) -> None:
        """Re-draw the current dart list at the new board geometry."""
        self._redraw_darts()

    def _on_drag_release(self, index: int, position: Position, inside: bool) -> None:
        """Handle a dart marker drag release, updating the model."""
        if index < 0 or index >= len(self._darts):
            return

        throw_result = self._darts[index]
        if throw_result.dart_throw.position is None and not inside:
            return

        self._cancel_debounce()
        self._debounce_after_id = self._view.after(
            DRAG_RELEASE_DEBOUNCE_MS,
            lambda tr=throw_result, pos=position: self._apply_dart_drag(tr, pos),
        )

    def _apply_dart_drag(
        self, throw_result: ThrowResult, new_position: Position
    ) -> None:
        self._debounce_after_id = None
        if self._model.game is None or throw_result.bust:
            return

        scoring_position = canvas_to_relative_position(new_position)
        scored = score_dart_throw(scoring_position)
        # TODO replace with model update call
        print(f"New throw: {scored} for throw result: {throw_result}")

    def _cancel_debounce(self) -> None:
        if self._debounce_after_id is not None:
            self._view.after_cancel(self._debounce_after_id)
            self._debounce_after_id = None

    def start(self) -> None:
        self._dartboard_view.set_resize_callback(self._on_board_resized)
        self._dartboard_view.set_drag_release_callback(self._on_drag_release)

    def _toggle_dartboard_visibility(self) -> None:
        self._left_view.set_dartboard_visible(self._dartboard_visible.get())
        self._view.refresh_left_panes()
