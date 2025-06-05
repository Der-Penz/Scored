import tkinter as tk
from scored.game.dartGame import DartGame


class PlayerView:
    def __init__(self, parent, game: DartGame):
        self.game = game
        self.frame = tk.Frame(parent)
        self.frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        self.__player_score_labels = {}

        self._build()

    def _build(self):
        num_players = len(self.game.players)
        if num_players == 0:
            return

        for index, player in enumerate(self.game.players):
            is_current_player = self.game.current_player == player

            tile = tk.Frame(
                self.frame,
                borderwidth=2,
                relief="solid",
                padx=10,
                pady=5,
                bg="lightblue" if is_current_player else None
            )
            tile.grid(row=0, column=index, sticky="nsew", padx=5)

            label = tk.Label(tile, text=player.name, font=("Arial", 12, "bold"), justify="center")
            label.pack()

            score_label = tk.Label(tile, text=f"Score: {player.set.current_leg.score}", font=("Arial", 11), justify="center")
            score_label.pack()

            self.__player_score_labels[player.name] = (score_label, tile)
            self.frame.grid_columnconfigure(index, weight=1)

    def update(self):
        for player in self.game.players:
            score_label, tile = self.__player_score_labels.get(player.name, (None, None))
            score_label.config(text=f"Score: {player.set.current_leg.score}")
            is_current = self.game.current_player == player
            tile.config(bg="lightblue" if is_current else "SystemButtonFace")
