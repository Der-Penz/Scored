import tkinter as tk
import ttkbootstrap as ttk
from tkinter import simpledialog, messagebox
from typing import Dict, List

from gui.protocols.controller_protocol import ControllerProtocol
from gui.model.model import AppModel
from gui.view.app_view import AppView

from scored_lib.game.player import Player
from scored_lib.game.dart_leg import DartLeg
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.game.rule import StartRule, FinishRule


class DartGameController(ControllerProtocol):
    def __init__(self, view: AppView, model: AppModel):
        super().__init__(view, model)
        self._app_view = view
        self._model = model
        self.view = self._app_view.game_view

        self._players: List[Player] = []
        self._legs: Dict[str, DartLeg] = {}
        self._current_index = 0

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
        self._app_view.master.bind_all("<Control-p>", lambda event: self._add_player())
        self._app_view.master.bind_all("<Control-g>", lambda event: self._start_game())

    def start(self) -> None:
        # nothing to start background-wise
        pass

    def _add_player(self) -> None:
        """Prompt for a player name and add them to the player list."""
        name = ttk.Querybox.get_string(
            prompt="Player name:",
            title="Add Player",
            parent=self._app_view,
        )
        if not name:
            return
        p = Player(name=name)
        self._players.append(p)
        self.view.set_players(self._players)

    def _remove_player(self) -> None:
        """Remove the last player from the list and update the view."""
        if not self._players:
            ttk.Messagebox.show_info(
                message="No players to remove.",
                title="Remove Player",
                parent=self._app_view,
            )
            return
        removed = self._players.pop()
        self.view.set_players(self._players)
        ttk.Messagebox.show_info(
            message=f"Removed {removed.name}",
            title="Remove Player",
            parent=self._app_view,
        )

    def _start_game(self) -> None:
        """Start a new game session after validating players and settings."""
        if not self._players:
            Messagebox.show_info(
                message="Add at least one player before starting.",
                title="Start Game",
                parent=self._app_view,
            )
            return

        settings = self._ask_game_settings()
        if settings is None:
            return
        starting_score, start_rule, finish_rule = settings

        # initialize legs
        self._legs = {
            p.id: DartLeg(starting_score, start_rule, finish_rule)
            for p in self._players
        }
        self._current_index = 0
        self._update_view_players()
        self._update_current_player()
        # clear turn slots
        self.view.set_turn_slots((None, None, None))

    def _ask_game_settings(self) -> tuple[int, StartRule, FinishRule] | None:
        """Open a modal dialog to configure game settings using ttkbootstrap.

        Returns
        -------
        tuple[int, StartRule, FinishRule] | None
            The starting score, start rule, and finish rule, or None if cancelled.
        """
        dialog = ttk.Toplevel(self._app_view)
        dialog.title("Game Settings")
        dialog.transient(self._app_view)
        dialog.grab_set()

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
            """Validate and save settings before destroying the dialog."""
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
            """Cancel dialog execution."""
            dialog.destroy()

        btnframe = ttk.Frame(dialog)
        btnframe.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(btnframe, text="OK", command=on_ok).pack(side="left", padx=6)
        ttk.Button(btnframe, text="Cancel", command=on_cancel).pack(side="left", padx=6)

        self._app_view.wait_window(dialog)

        if not result:
            return None
        return result["starting_score"], result["start_rule"], result["finish_rule"]

    def _update_view_players(self) -> None:
        self.view.set_players(self._players)
        for p in self._players:
            leg = self._legs.get(p.id)
            if leg:
                self.view.update_player_score(p.id, leg.score)

    def _update_current_player(self) -> None:
        if not self._players:
            return
        cur = self._players[self._current_index]
        self.view.set_current_player(cur.id)

    def add_throw(self, dart_throw: DartThrow) -> None:
        """External API: accept a DartThrow (from buttons or other sources)."""
        if not self._players or not self._legs:
            return
        player = self._players[self._current_index]
        leg = self._legs[player.id]
        result, end_turn = leg.add_throw(dart_throw)

        # update slot display
        # determine current turn throws texts
        last_turn = leg._results[-1] if leg._results else []
        texts = []
        for tr in last_turn:
            if tr is None:
                texts.append(None)
            else:
                texts.append(tr.dart_throw.short_label)

        # pad to 3
        while len(texts) < 3:
            texts.append(None)

        suggestion = None
        if leg.score > 0:
            suggestion = leg.get_best_checkout_path()

        self.view.set_turn_slots(tuple(texts[:3]), suggestion)
        self.view.update_player_score(player.id, leg.score)

        if result.finished:
            messagebox.showinfo("Game", f"{player.name} finished the leg!")
            return

        if end_turn:
            # show overlay with the three throws, then advance to next player
            def _after_overlay():
                self._current_index = (self._current_index + 1) % len(self._players)
                self._update_current_player()
                # clear turn slots for new player
                self.view.set_turn_slots((None, None, None))

            # show overlay for 900ms
            self.view.show_turn_overlay(
                tuple(texts[:3]), duration=900, on_hidden=_after_overlay
            )
