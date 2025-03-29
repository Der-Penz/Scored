import argparse
import os
from pathlib import Path
import time
from typing import List, Tuple

import cv2

from src.preparation.dataset.yolo import YoloAnnotations, YoloConfig, read_yolo_annotation
from src.board.dartboard import DartBoard
from src.predictor import DartPrediction, DartPredictor
from src.util import loading_bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate a model on a given dataset"
    )

    parser.add_argument(
        "model",
        type=str,
        help="path to the model to be evaluated",
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="path to the dataset to evaluate the model on",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="confidence threshold for the model"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="limit the number of images to evaluate on, -1 for no limit"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    dataset_path = Path(args.dataset)

    if not model_path.exists():
        raise FileNotFoundError(f"model file not found at {model_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset directory not found at {dataset_path}")
    try:
        config = YoloConfig.from_yaml(dataset_path)
    except Exception as e:
        print("could not read yolo config:")
        print(e)
        exit(1)

    if config.test is None:
        raise ValueError("test subset not found in config")
    
    test_path = config.test

    test_path_images = test_path
    test_path_labels = test_path.parent / "labels"

    
    
    print(f"| Available classes:")
    for c, points in config.classes.items():
        print(f"├──> {c} : {", ".join(points)}")
    print("-" * 30)

    model = DartPredictor(DartBoard(1000), model_path, args.conf)
    print(f"Start inference on test set: {test_path} with model {model_path}")

    files = os.listdir(test_path_images)

    if args.limit > 0:
        files = files[:args.limit]

    total = len(files)
    print(f"Running inference on {total} files:")
    inference_results : List[Tuple[YoloAnnotations, DartPrediction, float]] = []
    for i, image_path in enumerate(files):
        loading_bar(i, total)

        start_time = time.time()

        if not image_path.endswith(".jpg"):
            print(f"skipping non-image file {image_path}")
            continue
        image = cv2.imread(test_path_images / image_path)

        if image is None:
            print(f"could not read image at {image_path}")
            continue
        
        label_path = test_path_labels / f"{image_path.split(".")[0]}.txt"
        ground_truth = read_yolo_annotation(label_path, config.classes)
        
        prediction = model.predict(image, verbose=False)

        took = time.time() - start_time
        inference_results.append((ground_truth, prediction, took))
        
    loading_bar(total, total, newline=True)
    print("Inference finished")
    print("-" * 30)


    print(f"Total time: {sum([took for _, _, took in inference_results])} seconds")
    print(f"Average time: {sum([took for _, _, took in inference_results]) / len(inference_results)} seconds")







