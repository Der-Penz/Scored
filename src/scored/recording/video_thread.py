import tkinter as tk
import cv2
import threading
from PIL import Image, ImageTk


class VideoThread(threading.Thread):
    def __init__(self, source : str, video_label : tk.Label, root : tk.Tk):
        super().__init__()
        self.video_url = source
        self.video_label = video_label
        self.cap = cv2.VideoCapture(source)
        self.root = root
        self.running = True
        self.frame : Image = None

    def run(self):
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

        self.frame = frame

    def is_running(self):
        return self.running
    
    @property
    def current_frame(self):
        return self.frame

    def stop(self):
        print("Stopping video thread...")
        self.running = False
        self.cap.release()
