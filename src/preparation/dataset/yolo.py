from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import yaml

from preparation.dataset.annotation import YoloAnnotation
from util import BBYolo


@dataclass(frozen=True)
class YoloAnnotations:
    """
    A collection of YOLO annotations for different classes.
    """

    annotations: Dict[str, List[YoloAnnotation]]

    def get_class(self, class_name: str) -> List[YoloAnnotation]:
        """
        Get the annotations for a given class.

        :param class_name: The name of the class.
        :return: A list of YoloAnnotation objects for the given class.
        """
        return self.annotations.get(class_name, [])


@dataclass(frozen=True)
class YoloConfig:
    """
    A class to represent the yolo config.

    Attributes:
        base_dir (Path): The base directory of the dataset.
        train (Path): The path to the training set, None if not set.
        val (Path): The path to the validation set, None if not set.
        test (Path): The path to the test set, None if not set.
        classes (Dict[str, List[str]]): A dictionary containing the class names and their keypoints.
    """

    base_dir: Path
    train: Path
    val: Path
    test: Path
    classes: Dict[str, List[str]]

    def __post_init__(self):
        """
        Convert the paths to absolute paths.
        """
        object.__setattr__(
            self, "train", self.base_dir / self.train if self.train else None
        )
        object.__setattr__(self, "val", self.base_dir / self.val if self.val else None)
        object.__setattr__(
            self, "test", self.base_dir / self.test if self.test else None
        )

    @staticmethod
    def from_yaml(file_path: str | Path) -> YoloConfig:
        """
        Read the yolo config from the given path.

        :param dir_path: The path to the .yaml file.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        if file_path.suffix != ".yaml":
            raise ValueError(f"Config file is not a yaml file: {file_path}")

        with open(file_path, "r") as file:
            content = file.read()
            raw_config = yaml.safe_load(content)

            train_set = raw_config.get("train")
            val_set = raw_config.get("val")
            test_set = raw_config.get("test")

            classes = {}
            for class_name in raw_config["names"].values():
                classes[class_name] = raw_config["keypoint_names"][class_name]

            return YoloConfig(
                base_dir=file_path.parent,
                train=Path(train_set) if train_set else None,
                val=Path(val_set) if val_set else None,
                test=Path(test_set) if test_set else None,
                classes=classes,
            )

    def class_names(self) -> List[str]:
        """
        Get the class names from the config.

        :return: A list of class names.
        """
        return list(self.classes.keys())


def read_yolo_annotation(
    annotations_path: Path,
    classes: Dict[str, List[str]],
) -> YoloAnnotations:
    """
    Read a yolo .txt keypoint annotation file and return a list of YoloAnnotation objects.

    :param annotations_path: The path to the annotations file
    :param classes: A dictionary containing the class names and their keypoints
    :return: YoloAnnotations object containing the annotations
    """
    with open(annotations_path, "r") as file:
        lines = file.readlines()

    class_names = list(classes.keys())

    annotations = defaultdict(list)
    for line in lines:
        data = line.split(" ")
        class_index = int(data[0])
        class_name = class_names[class_index]
        bb = BBYolo((float(data[1]), float(data[2])), float(data[3]), float(data[4]))
        keypoints = []
        for i, keypoint_label in zip(range(5, len(data), 3), classes[class_name]):
            keypoints.append((float(data[i]), float(data[i + 1]), keypoint_label))

        annotations[class_name].append(
            YoloAnnotation(class_index, class_name, bb, keypoints)
        )
    return YoloAnnotations(annotations)
