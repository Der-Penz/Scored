import tkinter as tk

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import (
    DartThrowEvent,
    GameStarted,
    PlayerAdded,
    PlayerRemoved,
    ScoreChanged,
    ThrowEdited,
    TurnChanged,
)
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from scored_lib.game.game_leg import GameLeg
from scored_lib.game.player import Player
from scored_lib.game.rule import FinishRule, StartRule


class DartGameController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self.game_view = self._view.game_view
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
        self._event_channel.subscribe(ThrowEdited, self._on_throw_edited)
        self.game_view.scorepad_view.turn_overlay.bind_callbacks(
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

        _, throw_result, end_turn = self._model.game.add_throw(event.throw)

        self._event_channel.emit(ScoreChanged())

        current_leg = self._model.game.current_leg
        self.game_view.turn_view.set_throw(
            3 - current_leg.throws_left, str(throw_result.dart_throw.short_label)
        )

        if self._model.game.is_finished:
            winner = self._model.game.winner
            self.game_view.turn_view.reset_throws()
            ttk.Messagebox.show_info(
                message=f"{winner.name} wins!" if winner else "Game finished.",
                title="Game Over",
                parent=self._view,
            )
            return

        if end_turn:
            self._turn_pending = True
            self.game_view.scorepad_view.turn_overlay.show()

    def _on_throw_edited(self, event: ThrowEdited) -> None:
        current_leg = self._model.game.current_leg
        throw_result = current_leg.current_turn_throws[event.throw - 1]
        self.game_view.turn_view.set_throw(
            event.throw, str(throw_result.dart_throw.short_label)
        )

    def _on_turn_next(self) -> None:
        """Confirm the finished turn and advance to the next player."""
        self._turn_pending = False
        self._model.game.next_player()
        self.game_view.scorepad_view.turn_overlay.hide()
        self.game_view.turn_view.reset_throws()
        self._event_channel.emit(TurnChanged())

    def _on_turn_undo(self) -> None:
        """Undo the last throw and let the same player throw again."""
        self._turn_pending = False
        self._model.game.undo_last_throw()
        self.game_view.scorepad_view.turn_overlay.hide()
        self.game_view.turn_view.reset_throws()
        self._refresh_current_turn()
        self._event_channel.emit(ScoreChanged())

    def _refresh_current_turn(self) -> None:
        """Redraw the current player's registered throws into the turn view."""
        current_leg = self._model.game.current_leg
        for idx, throw_result in enumerate(current_leg.current_turn_throws):
            self.game_view.turn_view.set_throw(
                idx + 1, str(throw_result.dart_throw.short_label)
            )

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

        selected = self._ask_remove_player()
        if selected is None:
            return

        self._model.players.remove(selected)
        self._event_channel.emit(PlayerRemoved(player=selected))

    def _ask_remove_player(self) -> Player | None:
        """Show a modal dialog with a dropdown to pick which player to remove."""
        dialog = ttk.Toplevel(self._view)
        dialog.title("Remove Player")
        dialog.transient(self._view)
        dialog.grab_set()
        self._center_dialog(dialog)

        ttk.Label(dialog, text="Select player to remove:").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=6, pady=6
        )

        players_by_name = {p.name: p for p in self._model.players}
        names = list(players_by_name.keys())
        selected_var = tk.StringVar(value=names[0])
        combo = ttk.Combobox(
            dialog,
            textvariable=selected_var,
            values=names,
            state="readonly",
        )
        combo.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        result = {}

        def on_remove() -> None:
            name = selected_var.get()
            if name in players_by_name:
                result["player"] = players_by_name[name]
                dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        btnframe = ttk.Frame(dialog)
        btnframe.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(btnframe, text="Remove", command=on_remove).pack(side="left", padx=6)
        ttk.Button(btnframe, text="Cancel", command=on_cancel).pack(side="left", padx=6)

        self._view.wait_window(dialog)

        return result.get("player")

    def _center_dialog(self, dialog: tk.Toplevel) -> None:
        """Center a dialog over its parent window on screen."""
        dialog.update_idletasks()
        dialog_w, dialog_h = dialog.winfo_width(), dialog.winfo_height()

        parent = self._view
        try:
            x = parent.winfo_rootx()
            y = parent.winfo_rooty()
            parent_w = parent.winfo_width()
            parent_h = parent.winfo_height()
        except tk.TclError:
            x = y = 0
            parent_w = parent_h = 0

        x_pos = x + max(0, (parent_w - dialog_w) // 2)
        y_pos = y + max(0, (parent_h - dialog_h) // 2)
        dialog.geometry(f"+{x_pos}+{y_pos}")

    def _start_game(self) -> None:
        """Start a new game session after validating players and settings."""
        if not self._model.players:
            ttk.Messagebox.show_info(
                message="Add at least one player before starting.",
                title="Start Game",
                parent=self._view,
            )
            return

        settings = self._ask_game_settings()
        if settings is None:
            return
        starting_score, start_rule, finish_rule = settings

        self._model.game = GameLeg(
            players=tuple(self._model.players),
            starting_score=starting_score,
            start_rule=start_rule,
            finish_rule=finish_rule,
        )

        self.game_view.turn_view.reset_throws()

        self._event_channel.emit(GameStarted())

    def _ask_game_settings(self) -> tuple[int, StartRule, FinishRule] | None:
        """Open a modal dialog to configure game settings using ttkbootstrap."""
        dialog = ttk.Toplevel(self._view)
        dialog.title("Game Settings")
        dialog.transient(self._view)
        dialog.grab_set()
        self._center_dialog(dialog)

        ttk.Label(dialog, text="Starting Score:").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )
        start_entry = ttk.Entry(dialog)
        start_entry.insert(0, "501")
        start_entry.grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(dialog, text="Start Rule:").grid(
            row=1, column=0, sticky="w", padx=6, pady=6
        )
        start_var = tk.StringVar(value=StartRule.ANY.name)
        start_combo = ttk.Combobox(
            dialog,
            textvariable=start_var,
            values=[s.name for s in StartRule],
            state="readonly",
        )
        start_combo.grid(row=1, column=1, padx=6, pady=6)

        ttk.Label(dialog, text="Finish Rule:").grid(
            row=2, column=0, sticky="w", padx=6, pady=6
        )
        finish_var = tk.StringVar(value=FinishRule.DOUBLE.name)
        finish_combo = ttk.Combobox(
            dialog,
            textvariable=finish_var,
            values=[f.name for f in FinishRule],
            state="readonly",
        )
        finish_combo.grid(row=2, column=1, padx=6, pady=6)

        result = {}

        def on_ok() -> None:
            try:
                s = int(start_entry.get())
            except ValueError:
                ttk.Messagebox.show_error(
                    message="Starting score must be an integer.",
                    title="Invalid",
                    parent=dialog,
                )
                return
            result["starting_score"] = s
            result["start_rule"] = StartRule[start_var.get()]
            result["finish_rule"] = FinishRule[finish_var.get()]
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        btnframe = ttk.Frame(dialog)
        btnframe.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(btnframe, text="OK", command=on_ok).pack(side="left", padx=6)
        ttk.Button(btnframe, text="Cancel", command=on_cancel).pack(side="left", padx=6)

        self._view.wait_window(dialog)

        if not result:
            return None
        return result["starting_score"], result["start_rule"], result["finish_rule"]
