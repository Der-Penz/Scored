import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import (
    DartThrowEvent,
    GameStarted,
    PlayerAdded,
    PlayerRemoved,
    ScoreChanged,
    TurnChanged,
)
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from gui.view.widgets.dartgame_dialogs import ask_game_settings, ask_remove_player
from scored_lib.game.game_leg import GameLeg
from scored_lib.game.player import Player


class DartGameController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self.game_view = self._view.game_view
        self.turn_view = self._view.game_view.turn_view
        self.turn_overlay = self.game_view.scorepad_view.turn_overlay
        self._turn_pending = False

    def bind_menu(self, menu: ttk.Menu) -> None:
        menu.add_command(
            label="Add Player", command=self._add_player, accelerator="Ctrl+P"
        )
        menu.add_command(label="Remove Player", command=self._remove_player)
        menu.add_separator()
        menu.add_command(
            label="Start Game", command=self._start_game, accelerator="Ctrl+G"
        )

    def bind_components(self) -> None:
        self._view.master.bind_all("<Control-p>", lambda _: self._add_player())
        self._view.master.bind_all("<Control-g>", lambda _: self._start_game())
        self._event_channel.subscribe(DartThrowEvent, self._on_dart_throw)
        self._event_channel.subscribe(
            ScoreChanged, lambda _: self._refresh_current_turn()
        )
        self.turn_overlay.bind_callbacks(
            on_next=self._on_turn_next, on_undo=self._on_turn_undo
        )

    def _on_dart_throw(self, event: DartThrowEvent) -> None:
        if self._model.game is None:
            ttk.Messagebox.show_info(
                message="Start a game before throwing darts.",
                title="No Game Running",
                parent=self._view,
            )
            return
        if self._turn_pending:
            return

        _, _, end_turn = self._model.game.add_throw(event.throw)

        self._event_channel.emit(ScoreChanged())

        if self._model.game.is_finished:
            winner = self._model.game.winner
            self.turn_view.reset_throws()
            ttk.Messagebox.show_info(
                message=f"{winner.name} wins!" if winner else "Game finished.",
                title="Game Over",
                parent=self._view,
            )
            return

        if end_turn:
            self._turn_pending = True
            self.turn_overlay.show()

    def _on_turn_next(self) -> None:
        """Confirm the finished turn and advance to the next player."""
        if (
            self._turn_pending is False
        ):  # prevent accidental clicks when no turn is pending from focus issues
            return
        self._turn_pending = False

        self.turn_overlay.hide()
        self.turn_view.reset_throws()
        self._model.game.next_player()
        self._event_channel.emit(TurnChanged())

    def _on_turn_undo(self) -> None:
        """Undo the last throw and let the same player throw again."""
        if (
            self._turn_pending is False
        ):  # prevent accidental clicks when no turn is pending from focus issues
            return
        self._turn_pending = False

        self._model.game.undo_last_throw()
        self.turn_overlay.hide()
        self._event_channel.emit(ScoreChanged())

    def _refresh_current_turn(self) -> None:
        """Redraw the current player's registered throws into the turn view."""
        if self._model.game is None:
            return

        current_leg = self._model.game.current_leg
        self.turn_view.reset_throws()
        for idx, throw_result in enumerate(current_leg.current_turn_throws):
            self.turn_view.set_throw(idx + 1, str(throw_result.dart_throw.short_label))

    def start(self) -> None:
        pass

    def _add_player(self) -> None:
        """Prompt for a player name and add them to the player list."""
        name = ttk.Querybox.get_string(
            prompt="Player name:",
            title="Add Player",
            parent=self._view,
        )
        if not name:
            return
        p = Player(name=name)
        self._model.players.append(p)
        self._event_channel.emit(PlayerAdded(player=p))

    def _remove_player(self) -> None:
        """Prompt the user to select a player to remove, then update the view."""

        if self._model.game is not None:
            ttk.Messagebox.show_info(
                message="Cannot remove players while a game is in progress.",
                title="Remove Player",
                parent=self._view,
            )
            return

        if not self._model.players:
            ttk.Messagebox.show_info(
                message="No players to remove.",
                title="Remove Player",
                parent=self._view,
            )
            return

        selected = ask_remove_player(self._view, self._model.players)
        if selected is None:
            return

        self._model.players.remove(selected)
        self._event_channel.emit(PlayerRemoved(player=selected))

    def _start_game(self) -> None:
        """Start a new game session after validating players and settings."""
        if not self._model.players:
            ttk.Messagebox.show_info(
                message="Add at least one player before starting.",
                title="Start Game",
                parent=self._view,
            )
            return

        settings = ask_game_settings(self._view)
        if settings is None:
            return
        starting_score, start_rule, finish_rule = settings

        self._model.game = GameLeg(
            players=tuple(self._model.players),
            starting_score=starting_score,
            start_rule=start_rule,
            finish_rule=finish_rule,
        )

        self.turn_view.reset_throws()

        self._event_channel.emit(GameStarted())
