from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from scored.util import BBYolo, loading_bar


@dataclass(frozen=True)
class LabelStudioKeypoint:
    """
    A keypoint in the Label Studio format.

    Attributes:
        pos (Tuple[int, int]): The position of the keypoint
        label (str): The label of the keypoint
        id (str): The id of the keypoint
        parent_id (str): The id of the parent
    """

    pos: Tuple[int, int]
    label: str
    id: str
    parent_id: str = None


@dataclass(frozen=True)
class LabelStudioObject:
    """
    Represents an object in the Label Studio format.

    Attributes:
        label (str): The label of the object
        id (str): The id of the object
        bbox (BBYolo): The bounding box of the object
        keypoints (List[LabelStudioKeypoint]): The list of keypoints in the object
    """

    label: str
    id: str
    bbox: BBYolo
    keypoints: List[LabelStudioKeypoint]


@dataclass(frozen=True)
class LabelStudioAnnotation:
    """
    Represents an annotation in the Label Studio format.

    Attributes:
        id (int): The id of the annotation
        img_path (Path): The path to the image
        objects (List[LabelStudioObject]): The list of objects in the annotation
    """

    id: int
    img_path: Path
    objects: List[LabelStudioObject]

    def get_objects_of(self, label: str) -> List[LabelStudioObject]:
        """
        Returns the objects of the given label.

        :param label: The label to filter the objects by
        :return: A list of LabelStudioObject
        """
        return [obj for obj in self.objects if obj.label == label]


def extract_annotation_info(
    annotation_data: Dict[str, Any],
) -> List[LabelStudioObject]:
    """
    Extracts the keypoints and relations from the annotation data.

    :param annotation_data: The annotation data from Label Studio
    :return: A list of LabelStudioObject
    """

    keypoints: List[LabelStudioKeypoint] = []
    rects: List[any] = []

    for item in annotation_data:
        if item["type"] == "rectanglelabels":
            rects.append(item)

        elif item["type"] == "keypointlabels":
            label = item["value"]["keypointlabels"][0]
            position = item["value"]["x"] / 100, item["value"]["y"] / 100
            id = item["id"]
            parent = item["parentID"]
            keypoints.append(
                LabelStudioKeypoint(position, label.lower(), id, parent_id=parent)
            )

    objects: List[LabelStudioObject] = []
    for rect in rects:
        label = rect["value"]["rectanglelabels"][0]
        id = rect["id"]
        x1 = rect["value"]["x"]
        y1 = rect["value"]["y"]
        width = rect["value"]["width"]
        height = rect["value"]["height"]

        x1 = x1 / 100
        y1 = y1 / 100
        width = width / 100
        height = height / 100

        x1 = x1 + width / 2
        y1 = y1 + height / 2

        center = (x1, y1)
        bb = BBYolo(center, width, height)
        associated_keypoints: List[LabelStudioKeypoint] = [
            keypoint for keypoint in keypoints if keypoint.parent_id == id
        ]

        objects.append(LabelStudioObject(label.lower(), id, bb, associated_keypoints))

    return objects


def extract_annotations(
    annotations_path: Path, limit: int = 0
) -> List[LabelStudioAnnotation]:
    """
    Reads the annotations from the given file and extracts the keypoints positions and relations.

    :param annotations_path: The path to the annotations file
    :param limit: The maximum number of annotations to read, 0 means no limit
    :return: A list of annotations
    """

    with open(annotations_path, "r") as file:
        dataList: list = json.load(file)
    if limit != 0:
        dataList = dataList[:limit]

    total = len(dataList)
    annotations: List[LabelStudioAnnotation] = []
    for i, data in enumerate(dataList):
        if data is None:
            continue

        storage_path = data["data"]["img"]
        if not storage_path.startswith("/data/local-files/?d=dataset"):
            raise Exception(f"Invalid storage path : {storage_path}")

        storage_path = storage_path.replace("/data/local-files/?d=dataset", ".")
        basepath = Path(os.getenv("DATASET_PATH"))

        img_path = (basepath / storage_path).resolve()

        try:
            loading_bar(i, total)
            objects = extract_annotation_info(data["annotations"][0]["result"])

            annotations.append(LabelStudioAnnotation(data["id"], img_path, objects))
        except Exception as e:
            print(f"Error parsing entry {i + 1}. Entry id: {data['id']}")
            print(e)

    loading_bar(total, total)
    print()

    return annotations


def get_yolo_annotation_for_class(
    class_index: int,
    class_name: str,
    keypoints: List[str],
    annotation: LabelStudioAnnotation,
    max_keypoints: int,
    visible=True,
) -> List[List[str]]:
    """
    Generates the yolo annotation for the given annotation and class. If the keypoints are part of different objects, multiple entries are generated.

    :param class_index: The index of the class
    :param class_name: The name of the class
    :param keypoints: The list of keypoints for the class
    :param annotation: The annotation
    :param max_keypoints: The maximum number of keypoints over all classes, if the current class has less keypoints, the remaining ones are filled with zeros
    :param visible: Whether the keypoints are visible, only supports all visible or all invisible keypoints
    :return: A list of yolo annotations that can be written to a file
    """
    objects_of_class = annotation.get_objects_of(class_name)

    if len(objects_of_class) == 0:
        # if no objects are found, the class is not present in the annotation
        return []

    label_to_index = {label: index for index, label in enumerate(keypoints)}

    def get_annotation_line(bb: BBYolo, keypoints: List[LabelStudioKeypoint]):
        line = [class_index, bb.center[0], bb.center[1], bb.width, bb.height]

        for i in range(max_keypoints):
            if i >= len(keypoints):
                line.append(0)
                line.append(0)
                line.append(0)
                continue

            keypoint = keypoints[i]
            line.append(keypoint.pos[0])
            line.append(keypoint.pos[1])
            line.append(int(visible))

        return line

    lines = []
    for object in objects_of_class:
        selected_keypoints = [
            keypoint for keypoint in object.keypoints if keypoint.label in keypoints
        ]

        ordered_keypoints = sorted(
            selected_keypoints, key=lambda keypoint: label_to_index[keypoint.label]
        )

        if len(set([k.label for k in ordered_keypoints])) != len(keypoints):
            raise Exception(
                f"Missing keypoints in annotation. Needed: {keypoints}, available: {[keypoint.label for keypoint in ordered_keypoints]}"
            )

        lines.append(get_annotation_line(object.bbox, ordered_keypoints))

    return lines


@dataclass(frozen=True)
class YoloAnnotation:
    """
    Represents a yolo annotation.
    """

    class_id: int
    class_name: str
    bb: BBYolo
    keypoints: List[Tuple[int, int, str]]

    def keypoints_pos(self) -> List[Tuple[int, int]]:
        """
        Returns the keypoints positions.
        """
        return [(keypoint[0], keypoint[1]) for keypoint in self.keypoints]


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
