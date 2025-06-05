from queue import Queue
import tkinter as tk
import sv_ttk

from scored.board.dartboard import DartThrow, Multiplier
from scored.game.dartGame import DartGame, DartGameConfig
from scored.recording.views.player_view import PlayerView
from scored.recording.prediction_thread import PredictionThread
from scored.recording.video_thread import VideoThread
from scored.recording.views.throw_input_view import DartInputView


class App:
    def __init__(self, game: DartGame):
        self.game = game
        self.__root = tk.Tk()
        self.__root.title("Scored Recording")
        self.__root.geometry("800x600")

        #add a menu bar to start a new game and exit the app
        self.__menubar = tk.Menu(self.__root)
        self.__menubar.add_command(label="Exit", command=self.__root.quit)
        self.__root.config(menu=self.__menubar)

        self.__player_view = PlayerView(self.__root, self.game)
        self.__dart_input_view = DartInputView(self.__root, self.submit_score)

        #add a button that adds a throw to the current player's leg
        tk.Button(
            self.__root,
            text="Add score",
            command=lambda: self.__dart_input_view.add_throw(DartThrow(number=10, multiplier=Multiplier.SINGLE, position=None)),
            font=("Arial", 12)
        ).pack(side=tk.BOTTOM, pady=10)

    def submit_score(self):
        """
        Submits the score from the input view to the game.
        """
        throws = self.__dart_input_view.throws
        if not throws:
            print("No throws to submit.")
            return

        for throw in throws:
            if not self.game.current_player.set.current_leg.add_throw(throw):
                print(f"Throw {throw} did not close the leg.")
            else:
                print(f"Leg closed with throw {throw}.")

        self.game.next_player()
        self.__dart_input_view.clear_throws()
        self.__player_view.update()


    def get_root(self):
        return self.__root


def run_app(url: str, model_path: str, players: int):
    app = App(DartGame(DartGameConfig(num_players=players)))
    root = app.get_root()

    # video_label = tk.Label(root)
    # video_label.pack()

    # video_thread = VideoThread(url, video_label, root)
    # video_thread.daemon = True
    # video_thread.start()

    # try:
    #     frame_queue = Queue(3)
    #     result_queue = Queue(3)
    #     prediction_thread = PredictionThread(model_path, frame_queue, result_queue)
    #     prediction_thread.start()
    # except Exception as e:
    #     print(f"Error starting prediction thread: {e}")
    #     return

    # def send_frame():
    #     frame = video_thread.current_frame
    #     print("Capturing frame...")
    #     if frame is not None:
    #         print("Sending frame to prediction thread...")
    #         frame_queue.put(frame)

    # button = tk.Button(root, text="Capture", command=lambda: send_frame)

    # button.pack()

    def on_close():
        print("Closing...")
        # video_thread.stop()
        # video_thread.join(timeout=1)
        root.quit()

    # def poll_result_queue():
    #     try:
    #         result = result_queue.get_nowait()
    #         print(f"Received prediction result: {result}")
    #     except Exception:
    #         pass 

    #     root.after(100, poll_result_queue)

    # poll_result_queue()

    root.protocol("WM_DELETE_WINDOW", on_close)
    sv_ttk.set_theme("dark")
    root.mainloop()

    # if video_thread.is_running():
    #     print("Stopping video thread...")
    #     video_thread.join(timeout=1)

    print("Application closed.")
