import tkinter as tk
import sv_ttk

from scored.recording.video_thread import VideoThread

sv_ttk.set_dark_theme()


def create_app():
    root = tk.Tk()
    root.title("Scored Recording")
    root.geometry("800x600")

    return root


def run_app(url: str):
    root = create_app()

    video_label = tk.Label(root)
    video_label.pack()

    video_thread = VideoThread(url, video_label, root)
    video_thread.daemon = True
    video_thread.start()

    def on_close():
        print("Closing...")
        video_thread.stop()
        video_thread.join(timeout=1)
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    if video_thread.is_running():
        print("Stopping video thread...")
        video_thread.join(timeout=1)

    print("Application closed.")
