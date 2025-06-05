from pathlib import Path
import threading
from queue import Queue

from scored.board.dartboard import DartBoard
from scored.prediction.predictor import DartPrediction, DartPredictor


class PredictionThread(threading.Thread):
    def __init__(self, model_path : Path, frame_queue : Queue, result_queue : Queue):
        super().__init__()
        self.model = DartPredictor(DartBoard(1000), model_path, 0.8)
        self.frame_queue = frame_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            data = self.frame_queue.get()
            if data is None:
                break
            
            result = self.process_frame(data)
            if result is None:
                continue

            self.result_queue.put(result)

    def stop(self):
        self.frame_queue.put(None)
        self.join()

    def process_frame(self, frame) -> DartPrediction:
        print("Processing frame...")

        try:
            result = self.model.predict(frame)
            return result
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
