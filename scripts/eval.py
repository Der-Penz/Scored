import argparse
import os
from pathlib import Path
import time
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np

from scored.data_preparation.yolo import (
    YoloAnnotations,
    YoloConfig,
    read_yolo_annotation,
)
from scored.board.dartboard import DartBoard
from scored.prediction.predictor import DartPrediction, DartPredictor
from scored.test.compare import eval_prediction
from scored.util import loading_bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a model on a given dataset")

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
        "--conf", type=float, default=0.5, help="confidence threshold for the model"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="limit the number of images to evaluate on, -1 for no limit",
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="save the results to a csv file at this path, if not provided no csv will be saved",
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
        files = files[: args.limit]

    total = len(files)
    print(f"Running inference on {total} files:")
    inference_results: List[Tuple[YoloAnnotations, DartPrediction, float, str]] = []
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
        inference_results.append(
            (ground_truth, prediction, took, image_path.split(".")[0])
        )

    loading_bar(total, total, newline=True)
    print("Inference finished")
    print("-" * 30)

    print(
        f"Total time: {sum([took for _, _, took, _ in inference_results]):.2f} seconds"
    )
    print(
        f"Average time: {(sum([took for _, _, took, _ in inference_results]) / len(inference_results)):.2f} seconds"
    )

    eval = [eval_prediction(k[0], k[1], model._board) for k in inference_results]

    correct_count = 0
    for i, (result, (_, _, _, name)) in enumerate(zip(eval, inference_results)):
        if result.correct:
            correct_count += 1
            continue
        print(
            f"Image {name} : Incorrect cm: {result.confusion_matrix()} | (TP, FP, FN, TN)"
        )

    print("-" * 30)
    print(
        f"Accuracy: {correct_count / len(eval) * 100:.2f}% | Absolute: {correct_count}/{len(eval)}"
    )

    print("-" * 30)
    print("Per dart stats:")

    total_confusion_matrix = np.sum(
        [k.confusion_matrix() + (k.num_truth_darts,) for k in eval], axis=0
    )

    TP, FP, FN, TN, TOTAL = total_confusion_matrix

    accuracy = (TP + TN) / TOTAL * 100 if TOTAL > 0 else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(
        f"Confusion Matrix: \nTP: {total_confusion_matrix[0]} | FP: {total_confusion_matrix[1]} \nFN: {total_confusion_matrix[2]} | TN: {total_confusion_matrix[3]}"
    )

    print(f"Total darts: {total_confusion_matrix[4]}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1_score:.2f}%")

    if not args.csv:
        exit(0)

    data = []

    for res, (_, _, process_time, img_name) in tqdm(
        zip(eval, inference_results), total=len(eval)
    ):
        matrix = res.confusion_matrix()
        ground_truth_darts = res.num_truth_darts

        pred_positions = [dart.position for dart in res.pred]
        truth_positions = [dart.position for dart in res.truth]

        data.append(
            {
                "img_name": img_name,
                "TP": matrix[0],
                "FP": matrix[1],
                "FN": matrix[2],
                "TN": matrix[3],
                "num_darts": ground_truth_darts,
                "p_num_darts": len(res.pred),
                "process_time": process_time,
                "pred_positions": pred_positions,
                "truth_positions": truth_positions,
                "correct": res.correct,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(args.csv, index=False)
