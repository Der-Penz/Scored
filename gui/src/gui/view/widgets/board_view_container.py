import tkinter as tk
from math import ceil, sqrt
from gui.view.widgets.board_view import BoardView


class BoardViewsContainer(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.board_views: list[BoardView] = []
        self.current_view = 0
        self.grid_mode = False  # False: main+row, True: grid

        # Top bar (tabs + toggle button)
        self.top_bar = tk.Frame(self)
        self.top_bar.pack(side="top", fill="x")

        self.tabs_frame = tk.Frame(self.top_bar)
        self.tabs_frame.pack(side="left", fill="x", expand=True)

        self.toggle_button = tk.Button(
            self.top_bar, text="Grid View", command=self.toggleMode
        )
        self.toggle_button.pack(side="right")

        # Content area
        self.content = tk.Frame(self)
        self.content.pack(fill="both", expand=True)

    def toggleMode(self):
        self.grid_mode = not self.grid_mode
        self.toggle_button.config(text="Main+Row" if self.grid_mode else "Grid View")
        self.render_views()

    def add_view(self, board_view: BoardView):
        self.board_views.append(board_view)
        board_view.master = self.content
        self.render_views()

    def render_views(self):
        # Clear previous views
        for widget in self.content.winfo_children():
            widget.pack_forget()
            widget.grid_forget()

        n = len(self.board_views)
        if n == 0:
            return

        if self.grid_mode:
            # Dynamic grid size
            grid_size = ceil(sqrt(n))
            for idx, view in enumerate(self.board_views):
                r = idx // grid_size
                c = idx % grid_size
                view.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)

            # Make rows and columns expand equally
            for i in range(grid_size):
                self.content.grid_rowconfigure(i, weight=1)
                self.content.grid_columnconfigure(i, weight=1)

        else:
            # Main + row layout
            main_view = self.board_views[self.current_view]
            main_view.pack(side="top", fill="both", expand=True, padx=5, pady=5)

            if n > 1:
                row_frame = tk.Frame(self.content)
                row_frame.pack(side="top", fill="x", expand=False)

                for idx, view in enumerate(self.board_views):
                    if idx == self.current_view:
                        continue
                    view.pack(
                        in_=row_frame,
                        side="left",
                        expand=True,
                        fill="both",
                        padx=5,
                        pady=5,
                    )
