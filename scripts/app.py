import argparse
from pathlib import Path

from scored.board.dartboard import DartThrow
from scored.game.dartGame import DartGame, DartGameConfig
from scored.recording.ui import run_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a document in an image from the given video stream."
    )

    parser.add_argument("--players", type=int, default=1, help="Number of players.")
    parser.add_argument("url", type=str, help="HTTP URL of the camera stream.")
    parser.add_argument(
        "--model", type=str, default="default_model", help="Path to the model file."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if not args.url.startswith("http://") and not args.url.startswith("https://"):
        print("Invalid URL. Please provide a valid HTTP/HTTPS URL.")
        exit(1)

    try:
        run_app(args.url, Path(args.model), args.players)
    except Exception as e:
        print("Exception occurred while running the app:")
        print(e)
        exit(1)
