import argparse
import tkinter as tk

from gui.drawing.dartboard import draw_dartboard
from gui.drawing.debug import draw_alignment_lines
from gui.util import to_relative_coordinates
from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.scoring import score_dart_throw


throws: list[DartThrow] = []


def handle_click(event: tk.Event) -> None:
    """
    Calculates and prints the normalized coordinates (0 to 1) of a click.

    Parameters
    ----------
    event : tk.Event
        The click event containing the x and y coordinates.
    """
    rel_x, rel_y = to_relative_coordinates(event)

    throw = score_dart_throw((rel_x, rel_y))

    print(f"Clicked at: ({rel_x}) -> Scored: {throw.label}")
    throws.append(throw)
    event.widget.event_generate("<Configure>")


def draw_throws(canvas: tk.Canvas, size: float) -> None:
    """Draw all dart throws on the board (0,0 is center)."""
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    center_x = canvas_width / 2
    center_y = canvas_height / 2
    radius = size / 2

    for throw in throws:
        rel_x, rel_y = throw.position  # relative to center (0,0)

        norm = (rel_x**2 + rel_y**2) ** 0.5
        if norm > 1:
            rel_x /= norm
            rel_y /= norm

        # Convert to canvas coordinates
        x = center_x + rel_x * radius
        y = center_y + rel_y * radius

        r = 5  # radius of hit marker
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="white")


def main() -> None:
    """
    Main entry point to initialize the Tkinter UI and bindings.
    """
    parser = argparse.ArgumentParser(description="Dartboard visualization and scoring UI.")
    parser.add_argument("--width", type=int, default=600, help="Window width (default: 600)")
    parser.add_argument("--height", type=int, default=600, help="Window height (default: 600)")
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Interactive Dartboard")

    canvas = tk.Canvas(
        root,
        width=args.width,
        height=args.height,
        bg="#2e2e2e",
        highlightthickness=0,
        borderwidth=0,
    )
    canvas.pack(fill="both", expand=True)

    canvas.bind("<Button-1>", handle_click)

    def redraw(event: tk.Event | None = None) -> None:
        canvas.delete("all")
        size = min(canvas.winfo_width(), canvas.winfo_height()) * 0.9
        draw_dartboard(canvas, size)
        draw_alignment_lines(canvas)
        draw_throws(canvas, size)

    # Use a small delay on resize to prevent laggy redraws
    canvas.bind("<Configure>", lambda e: canvas.after(50, redraw))

    root.mainloop()


if __name__ == "__main__":
    main()
