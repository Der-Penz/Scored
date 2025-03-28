import os
from pathlib import Path
from typing import Dict, List

import yaml

from src.preparation.dataset.annotation import LabelStudioAnnotation, YoloAnnotation
from src.util import BBYolo


def read_yolo_config(file_path):
    """
    Read the yolo config from the given path.

    :param dir_path: The path to the .yaml file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Config file is not a file: {file_path}")
    
    if file_path.suffix != ".yaml":
        raise ValueError(f"Config file is not a yaml file: {file_path}")

    with open(file_path, "r") as file:
        content = file.read()
        raw_config = yaml.safe_load(content)
        config = {}

        config["subsets"] = [
            (k, v)
            for k, v in raw_config.items()
            if k in ["train", "val", "test"] and v is not None
        ]
        classes = {}
        for class_name in raw_config["names"].values():
            classes[class_name] = raw_config["keypoint_names"][class_name]
        config["classes"] = classes
        return config
    
def read_yolo_annotation(
    annotations_path: Path,
    classes: List[str],
    keypoints_per_class: Dict[str, List[str]],
) -> List[LabelStudioAnnotation]:
    """
    Read a yolo .txt keypoint annotation file and return a list of YoloAnnotation objects.

    :param annotations_path: The path to the annotations file
    :param classes: The list of class names
    :param keypoints_per_class: The dictionary containing the keypoints for each class
    :return: A list of YoloAnnotation
    """
    with open(annotations_path, "r") as file:
        lines = file.readlines()

    annotations: List[YoloAnnotation] = []
    for line in lines:
        data = line.split(" ")
        class_index = int(data[0])
        class_name = classes[class_index]
        bb = BBYolo((float(data[1]), float(data[2])), float(data[3]), float(data[4]))
        keypoints = []
        for i, keypoint_label in zip(
            range(5, len(data), 3), keypoints_per_class[class_name]
        ):
            keypoints.append((float(data[i]), float(data[i + 1]), keypoint_label))

        annotations.append(YoloAnnotation(class_index, class_name, bb, keypoints))
    return annotations
