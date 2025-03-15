from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from util import BBYolo, compute_bounding_box, loading_bar


@dataclass(frozen=True)
class LabelStudioKeypoint:
    """
    Represents a keypoint in the Label Studio format.
    Contains the position and the label of the keypoint and its id.
    """

    pos: Tuple[int, int]
    label: str
    id: str


@dataclass(frozen=True)
class LabelStudioRelation:
    """
    Represents a relation between two keypoints in the Label Studio format.
    Contains the ids of the from and to keypoints.
    """

    from_id: str
    to_id: str


@dataclass(frozen=True)
class LabelStudioAnnotation:
    """
    Represents an annotation for an image in the Label Studio format.
    Contains the image name, id, the keypoints and relations between them.
    """

    id: int
    img_name: str
    keypoints: List[LabelStudioKeypoint]
    relations: List[LabelStudioRelation]


def extract_annotation_info(
    annotation_data: Dict[str, Any],
) -> Tuple[List[LabelStudioKeypoint], List[LabelStudioRelation]]:
    """
    Extracts the keypoints and relations from the given annotation data.

    :param annotation_data: The annotation data in the Label Studio json format
    :return: A tuple containing the keypoints and relations
    """

    keypoints: List[LabelStudioKeypoint] = []
    relations: List[LabelStudioRelation] = []

    for item in annotation_data:
        if item["type"] == "relation":
            from_id = item["from_id"]
            to_id = item["to_id"]
            relations.append(LabelStudioRelation(from_id, to_id))

        elif item["type"] == "keypointlabels":
            label = item["value"]["keypointlabels"][0]
            position = item["value"]["x"] / 100, item["value"]["y"] / 100
            id = item["id"]
            keypoints.append(LabelStudioKeypoint(position, label.lower(), id))

    return keypoints, relations


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

        img_path = data["data"]["img"]

        if "dartboard" in img_path:
            img_name = "dartboard" + img_path.split("dartboard", 1)[-1]
        else:
            print("Img names must contain the string dartboard")
            continue
        try:
            loading_bar(i, total)
            keypoints, relations = extract_annotation_info(
                data["annotations"][0]["result"]
            )

            annotations.append(
                LabelStudioAnnotation(data["id"], img_name, keypoints, relations)
            )
        except Exception as e:
            print(f"Error parsing entry {i + 1}. Entry id: {data['id']}")
            print(e)

    loading_bar(total, total)
    print()

    return annotations


def get_yolo_annotation_for_class(
    class_index: int,
    keypoints: List[str],
    annotation: LabelStudioAnnotation,
    max_keypoints: int,
    visible=True,
    **kwargs,
) -> List[List[Any]]:
    """
    Generates the yolo annotation for the given annotation and class. If the keypoints are part of different objects, multiple entries are generated.

    :param class_index: The index of the class
    :param keypoints: The list of keypoints
    :param annotation: The annotation
    :param max_keypoints: The maximum number of keypoints over all classes, if the current class has less keypoints, the remaining ones are filled with zeros
    :param visible: Whether the keypoints are visible, only supports all visible or all invisible keypoints
    :param kwargs: Additional arguments for the compute_bounding_box function
    :return: A list of yolo annotations that can be written to a file
    """
    selected_keypoints = [
        keypoint for keypoint in annotation.keypoints if keypoint.label in keypoints
    ]

    # if no keypoints are found, the class is not present in the annotation
    if len(selected_keypoints) == 0:
        return []

    if len(set([k.label for k in selected_keypoints])) != len(keypoints):
        raise Exception(
            f"Missing keypoints in annotation. Needed: {keypoints}, available: {[keypoint.label for keypoint in selected_keypoints]}"
        )

    selected_keypoint_ids = [keypoint.id for keypoint in selected_keypoints]
    selected_relations = [
        relation
        for relation in annotation.relations
        if relation.from_id in selected_keypoint_ids
        or relation.to_id in selected_keypoint_ids
    ]
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

    # if no relations are found, all existing keypoints correspond to the same object
    if len(selected_relations) == 0:
        bb = compute_bounding_box([k.pos for k in selected_keypoints], **kwargs)
        ordered_keypoints = sorted(
            selected_keypoints, key=lambda keypoint: label_to_index[keypoint.label]
        )

        return [get_annotation_line(bb, ordered_keypoints)]

    lines = []

    # if relations are found, the keypoints are part of different objects so multiple entries are needed
    for relation in selected_relations:
        selected_keypoints_for_object = [
            keypoint
            for keypoint in selected_keypoints
            if keypoint.id in [relation.from_id, relation.to_id]
        ]
        bb = compute_bounding_box(
            [k.pos for k in selected_keypoints_for_object], **kwargs
        )
        ordered_keypoints = sorted(
            selected_keypoints_for_object,
            key=lambda keypoint: label_to_index[keypoint.label],
        )
        lines.append(get_annotation_line(bb, ordered_keypoints))
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
