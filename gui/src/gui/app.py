import tkinter as tk

from gui.widgets.label_view import LabelView
from gui.widgets.board_view_container import BoardViewsContainer


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Split Window App")
        self.root.geometry("600x400")

        # Configure grid to split window into two halves
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left frame
        self.left_frame = tk.Frame(self.root, bg="grey")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        # Right frame
        self.right_frame = tk.Frame(self.root, bg="grey")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.board_container = BoardViewsContainer(self.left_frame)
        self.board_container.pack(fill="both", expand=True)

        # Create board views with container as parent
        boards = [
            LabelView(self.board_container, "Classic"),
            LabelView(self.board_container, "Training"),
        ]

        # Give boards to container
        for view in boards:
            self.board_container.add_view(view)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
