import tkinter as tk
from tkinter import messagebox

from gui.events.event_channel import EventChannel
from gui.events.event_types import DartThrowEvent, GameStartedEvent
import ttkbootstrap as ttk

from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView
from scored_lib.game.dart_leg import DartLeg
from scored_lib.game.player import Player
from scored_lib.game.rule import FinishRule, StartRule


class DartGameController(BaseController):
    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self.game_view = self._view.game_view

    def bind_menu(self, menu: ttk.Menu) -> None:
        menu.add_command(label="Add Player", command=self._add_player, accelerator="Ctrl+P")
        menu.add_command(label="Remove Player", command=self._remove_player)
        menu.add_separator()
        menu.add_command(label="Start Game", command=self._start_game, accelerator="Ctrl+G")

    def bind_components(self) -> None:
        self._view.master.bind_all("<Control-p>", lambda _: self._add_player())
        self._view.master.bind_all("<Control-g>", lambda _: self._start_game())
        self._event_channel.subscribe(DartThrowEvent, self._on_dart_throw)

    def _on_dart_throw(self, _event: DartThrowEvent) -> None:
        pass

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
        self.game_view.set_players(self._model.players)

    def _remove_player(self) -> None:
        """Remove the last player from the list and update the view."""
        if not self._model.players:
            ttk.Messagebox.show_info(
                message="No players to remove.",
                title="Remove Player",
                parent=self._view,
            )
            return
        removed = self._model.players.pop()
        self.game_view.set_players(self._model.players)
        ttk.Messagebox.show_info(
            message=f"Removed {removed.name}",
            title="Remove Player",
            parent=self._view,
        )

    def _start_game(self) -> None:
        """Start a new game session after validating players and settings."""
        if not self._model.players:
            messagebox.show_info(
                message="Add at least one player before starting.",
                title="Start Game",
                parent=self._view,
            )
            return

        settings = self._ask_game_settings()
        if settings is None:
            return
        starting_score, start_rule, finish_rule = settings

        self._model.legs = {
            p.id: DartLeg(starting_score, start_rule, finish_rule) for p in self._model.players
        }
        self._model.current_player_index = 0
        
        self._event_channel.publish(GameStartedEvent())

    def _ask_game_settings(self) -> tuple[int, StartRule, FinishRule] | None:
        """Open a modal dialog to configure game settings using ttkbootstrap."""
        dialog = ttk.Toplevel(self._view)
        dialog.title("Game Settings")
        dialog.transient(self._view)
        dialog.grab_set()

        ttk.Label(dialog, text="Starting Score:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        start_entry = ttk.Entry(dialog)
        start_entry.insert(0, "501")
        start_entry.grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(dialog, text="Start Rule:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        start_var = tk.StringVar(value=StartRule.ANY.name)
        start_combo = ttk.Combobox(
            dialog,
            textvariable=start_var,
            values=[s.name for s in StartRule],
            state="readonly",
        )
        start_combo.grid(row=1, column=1, padx=6, pady=6)

        ttk.Label(dialog, text="Finish Rule:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
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