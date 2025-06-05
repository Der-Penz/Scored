import tkinter as tk
from typing import List

from scored.board.dartboard import DartThrow


class DartInputView:
    def __init__(self, parent, submit_callback):
        self.__score_labels = []
        self.__dart_throws: List[DartThrow] = []
        self.frame = tk.Frame(parent)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        self.submit_callback = submit_callback

        self._build()

    def _build(self):
        for i in range(3):
            tile = tk.Frame(self.frame, borderwidth=2, relief="solid", padx=10, pady=10)
            tile.grid(row=0, column=i, padx=10, sticky="nsew")

            label = tk.Label(
                tile, text=f"Dart {i + 1}", font=("Arial", 12, "bold"), justify="center"
            )
            label.pack()

            score_label = tk.Label(
                tile, text="", font=("Arial", 11), justify="center"
            )
            score_label.pack()
            self.__score_labels.append(score_label)

        # Submit button as 4th column
        submit_frame = tk.Frame(self.frame, padx=10)
        submit_frame.grid(row=0, column=3, sticky="nsew", padx=10)

        submit_button = tk.Button(
            submit_frame, text="Send Score", command=self.submit_callback
        )
        submit_button.pack(expand=True, fill=tk.BOTH)

        self.frame.grid_columnconfigure(3, weight=1)

    def _update(self):
        """
        Updates the input view with the current throws.
        """
        for i in range(3):
            throw = self.__dart_throws[i] if i < len(self.__dart_throws) else None
            if throw:
                score_text = throw.short_label
            else:
                score_text = ""


            self.__score_labels[i].config(text=score_text)
            

    def add_throw(self, throw: DartThrow):
        """
        Adds a throw to the input view.
        """
        if len(self.__dart_throws) < 3:
            self.__dart_throws.append(throw)
            self._update()

    @property
    def throws(self):
        """
        Returns the list of throws.
        """
        return self.__dart_throws

    def clear_throws(self):
        """
        Clears the list of throws.
        """
        self.__dart_throws = []
        self._update()
