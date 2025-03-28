import argparse
import os
from pathlib import Path

import cv2

from src.preparation.dataset.yolo import read_yolo_config, read_yolo_annotation
from src.board.dartboard import DartBoard
from src.predictor import DartPredictor
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

    args = parser.parse_args()

    model_path = Path(args.model)
    dataset_path = Path(args.dataset)

    if not model_path.exists():
        raise FileNotFoundError(f"model file not found at {model_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset directory not found at {dataset_path}")
    try:
        config = read_yolo_config(dataset_path)
    except Exception as e:
        print("could not read yolo config:")
        print(e)
        exit(1)

    test_subset = [subset for subset in config["subsets"] if subset[0] == "test"]
    if test_subset is None:
        raise ValueError("test subset not found in config")
    
    test_subset = test_subset[0]
    test_path = dataset_path.parent / test_subset[1]

    test_path_images = test_path
    test_path_labels = test_path.parent / "labels"

    classes = config["classes"]
    
    print(f"Reading test dir: {test_path}")
    print(f"Available classes:")
    for c, points in classes.items():
        print(f"{c} : {", ".join(points)}")

    
    # model = DartPredictor(DartBoard(1000), model_path, args.conf)
    files = os.listdir(test_path_images)
    total = len(files)
    for i, image_path in enumerate(files):
        loading_bar(i, total)
        if not image_path.endswith(".jpg"):
            print(f"skipping non-image file {image_path}")
            continue
        image = cv2.imread(test_path_images / image_path)

        if image is None:
            print(f"could not read image at {image_path}")
            continue
        
        label_path = test_path_labels / f"{image_path.split(".")[0]}.txt"
        ground_truth = read_yolo_annotation(label_path, list(classes.keys()), classes)
        
        # print(label)
        # throws = model.predict(image)
    loading_bar(total, total)





