import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Sequence
from src.preparation.dataset.annotation import (
    LabelStudioAnnotation,
    get_yolo_annotation_for_class,
)
from src.util import loading_bar


def generate_config(
    out: Path,
    splitted_data: Dict[str, Sequence[Any]],
    classes_with_keypoints: Dict[str, List[str]],
):
    """
    Generates a dataset.yaml file for the given data.

    :param out: The output directory
    :param splitted_data: The splitted data
    :param classes_with_keypoints: A dictionary mapping class names to their list of keypoints
    """
    config = []

    config.append(f"path: {str(out)}")

    for label, data in splitted_data.items():
        if len(data) == 0:
            config.append(f"{label}: #empty")
        else:
            config.append(f"{label}: {label}/images")

    config.append("")
    config.append(f"nc: {len(classes_with_keypoints)}")
    config.append(f"names:")
    for i, class_name in enumerate(classes_with_keypoints.keys()):
        config.append(f"  {i}: {class_name}")
    config.append("")

    max_keypoints = len(max(classes_with_keypoints.values(), key=lambda x: len(x)))
    config.append(f"nkpt: {max_keypoints}")
    config.append(f"kpt_shape: [{max_keypoints}, 3]")
    config.append(f"keypoint_names:")
    for class_name, keypoints in classes_with_keypoints.items():
        config.append(f"  {class_name}: {str(keypoints)}")

    with open(out / "dataset.yaml", "w") as file:
        for line in config:
            file.write(line + "\n")


def generate_file_structure(
    annotation_data: List[LabelStudioAnnotation],
    classes_with_keypoints: Dict[str, List[str]],
    out: Path,
    img_dir: Path,
    copy_image: bool = True,
    clear_existing: bool = False,
    **kwargs,
):
    """
    Generates the file structure for the given annotations.

    :param annotation_data: The annotations
    :param classes_with_keypoints: A dictionary mapping class names to their list of keypoints
    :param out: The output directory
    :param img_dir: The directory containing the images
    :param copy_image: Whether to copy the images to the output directory
    :param clear_existing: Whether to clear the existing output directory
    :param kwargs: Additional arguments for the get_annotation_in_yolo_format function
    """
    if not os.path.exists(out):
        os.makedirs(out)
        print(f"Directory {out} created.")
    else:
        print(f"Directory {out} already exists.")
        if clear_existing:
            shutil.rmtree(out)
            os.makedirs(out)

    total = len(annotation_data)

    img_out = out / "images"
    annotation_out = out / "labels"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(annotation_out, exist_ok=True)

    for i, annotation in enumerate(annotation_data):
        loading_bar(i, total)
        img_path = img_dir / annotation.img_name

        if not img_path.exists():
            print("missing image file" + str(img_path))
        elif copy_image:
            shutil.copy(img_path, img_out / annotation.img_name)

        annotation_out_file = annotation_out / (
            annotation.img_name.split(".")[0] + ".txt"
        )

        collected_lines = []
        for i, keypoints in enumerate(classes_with_keypoints.values()):
            try:
                yolo_data = get_yolo_annotation_for_class(
                    i, keypoints, annotation, **kwargs
                )
                collected_lines.extend(yolo_data)
            except Exception as e:
                print(f"\nError: {str(e)} for annotation id: {annotation.id}")

        with open(annotation_out_file, "w") as file:
            for i, line in enumerate(collected_lines):
                file.write(" ".join(map(lambda x: str(x), line)))
                if i < len(collected_lines) - 1:
                    file.write("\n")

    loading_bar(total, total)
    print()
