import tkinter as tk

import ttkbootstrap as ttk

from gui.helper import center_dialog
from scored_lib.game.player import Player
from scored_lib.game.rule import FinishRule, StartRule


def ask_remove_player(parent: tk.Misc, players: list[Player]) -> Player | None:
    """Show a modal dialog with a dropdown to pick which player to remove."""
    dialog = ttk.Toplevel(parent)
    dialog.title("Remove Player")
    dialog.transient(parent)
    dialog.grab_set()
    center_dialog(dialog, parent)

    ttk.Label(dialog, text="Select player to remove:").grid(
        row=0, column=0, columnspan=2, sticky="w", padx=6, pady=6
    )

    players_by_name = {p.name: p for p in players}
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

    parent.wait_window(dialog)

    return result.get("player")


def ask_game_settings(parent: tk.Misc) -> tuple[int, StartRule, FinishRule] | None:
    """Open a modal dialog to configure game settings using ttkbootstrap."""
    dialog = ttk.Toplevel(parent)
    dialog.title("Game Settings")
    dialog.transient(parent)
    dialog.grab_set()
    center_dialog(dialog, parent)
    dialog.focus_force()

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
    ttk.Button(btnframe, text="OK", command=on_ok, bootstyle="primary").pack(
        side="left", padx=6
    )
    ttk.Button(btnframe, text="Cancel", command=on_cancel).pack(side="left", padx=6)

    parent.wait_window(dialog)

    if not result:
        return None
    return result["starting_score"], result["start_rule"], result["finish_rule"]
