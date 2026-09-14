import tkinter as tk

import ttkbootstrap as ttk
from ttkbootstrap import Tableview

from gui.helper import center_dialog
from scored_lib.game.dart_leg import DartLeg
from scored_lib.game.game_leg import GameLeg
from scored_lib.game.player import Player


class ScorecardView(ttk.Toplevel):
    """
    Snapshot scorecard window showing each player's throw history for the
    current game. The data is captured at construction time and never updated
    live; reopen the window to see fresh data.
    """

    def __init__(
        self,
        master: tk.Misc,
        players: list[Player],
        game: GameLeg | None,
    ) -> None:
        super().__init__(master)
        self.title("Scorecard")
        self.transient(master)
        self.geometry("580x420")
        self.minsize(420, 280)
        center_dialog(self, self.master)

        self._tables: dict[Player, Tableview] = {}

        if not players:
            ttk.Label(self, text="No players to display.").pack(expand=True)
            return

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=8)

        for player in players:
            tab = ttk.Frame(self._notebook)
            self._notebook.add(tab, text=player.name)

            leg = game.leg_for(player) if game is not None else None
            table = self._create_table(tab, leg)
            self._tables[player] = table
            table.pack(fill="both", expand=True)

    def _create_table(self, master: tk.Misc, leg: DartLeg | None) -> Tableview:
        colors = ttk.Style().colors
        return Tableview(
            master=master,
            coldata=[
                {"text": "Round", "anchor": "center", "stretch": True},
                {"text": "Dart 1", "anchor": "center", "stretch": True},
                {"text": "Dart 2", "anchor": "center", "stretch": True},
                {"text": "Dart 3", "anchor": "center", "stretch": True},
                {"text": "Total", "anchor": "center", "stretch": True},
                {"text": "Remaining", "anchor": "center", "stretch": True},
            ],
            rowdata=self._build_rows(leg) if leg is not None else [],
            autofit=True,
            stripecolor=(colors.light, colors.bg),
        )

    def _build_rows(self, leg: DartLeg) -> list[tuple]:
        """Convert a leg's rounds into table rows (one row per round)."""
        rows = []
        for round_idx, round in enumerate(leg.rounds, start=1):
            if not round:
                continue
            throws = []
            total = 0
            remaining = None

            for result in round:
                if result is None:
                    throws.append("-")
                    continue
                throws.append(result.dart_throw.short_label)
                total += result.dart_throw.score
                remaining = result.score_after
                if result.bust:
                    total = 0

            throws.extend("-" for _ in range(3 - len(round)))
            if remaining is None:
                remaining = leg.score

            rows.append((round_idx, throws[0], throws[1], throws[2], total, remaining))
        return rows
