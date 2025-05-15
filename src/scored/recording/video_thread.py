import tkinter as tk
import cv2
import threading
from threading import Lock
from PIL import Image, ImageTk


class VideoThread(threading.Thread):
    def __init__(self, source : str, video_label : tk.Label, root : tk.Tk):
        super().__init__()
        self.video_url = source
        self.video_label = video_label
        self.source = source
        self.cap = None
        self.root = root
        self.running = True
        self.frame : Image = None
        self.frame_lock = Lock()

    def run(self):
        self.video_label.configure(text="Loading video stream...")
        
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"Failed to open video source: {self.source}")
            self.video_label.configure(text="Unable to open video stream.")
            return
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not self.cap.isOpened():
                print("Failed to open video stream.")
                break

            if not ret:
                print("Failed to capture frame from video stream.")
                break
            
            self.root.after(0, self.update_image, frame)

        if self.cap.isOpened():
            self.cap.release()

    def update_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        with self.frame_lock:
            self.frame = frame

    def is_running(self):
        return self.running
    
    @property
    def current_frame(self):
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        print("Stopping video thread...")
        self.running = False
        self.cap.release()
