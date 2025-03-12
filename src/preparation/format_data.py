import os
from pathlib import Path
from typing import Dict, List
import argparse
import shutil

from src.preparation.dataset.annotation import extract_annotations
from src.preparation.dataset.format import generate_config, generate_file_structure
from src.preparation.dataset.split import split_data


def is_valid_directory(path):
    """Check if the given path is a valid directory."""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Invalid directory: {path}")
    return path


def is_valid_json(path):
    """Check if the given path is a valid JSON file."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")

    if not path.lower().endswith(".json"):
        raise argparse.ArgumentTypeError(
            f"Invalid file type: {path} (expected a .json file)"
        )

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert label studio json data annotation format into yolo format"
    )
    parser.add_argument(
        "annotation_file", type=is_valid_json, help="The json annotation file"
    )
    parser.add_argument(
        "--img-dir",
        type=is_valid_directory,
        default="./",
        help="The location of the image files relative to the annotation file (default: same directory)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./train/data",
        help="The output directory where the files are saved",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding in percentage of the img size around the bounding box",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only take the first n annotations, 0 for all",
    )
    parser.add_argument(
        "--keypoint-order",
        type=str,
        nargs="*",
        action="append",
        required=True,
        help="The keypoints in the order they will appear in the generated files",
    )

    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        help="The classes for the different keypoints, used in combination if multiple keypoint_order are provided",
    )

    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[0.7, 0.0, 0.3],
        help="Proportions for splitting the dataset into train, val, and test sets. Must sum to 1.0",
    )

    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Do not copy the images into the output dataset",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the data before splitting",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator",
    )

    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear the existing output directory, if labels or images are already present in the output directory the files will be deleted, "
        "if not passed everything will be appended or overwritten",
    )

    parser.add_argument("--zip", action="store_true", help="zip the resulting dataset")

    args = parser.parse_args()

    if sum(args.split) != 1:
        parser.error("--split must add up to 1.")

    if args.classes and not args.keypoint_order:
        parser.error("--classes can only be used if --keypoint_order is provided.")

    if args.keypoint_order and args.classes:
        if len(args.keypoint_order) != len(args.classes) and len(args.classes) != 0:
            parser.error(
                "The number of --classes must match the number of --keypoint_order provided."
            )
    else:
        args.classes = list(range(len(args.keypoint_order)))
    classes: Dict[str, List[str]] = {
        class_name.lower(): [keypoint.lower() for keypoint in keypoints]
        for class_name, keypoints in zip(args.classes, args.keypoint_order)
    }
    annotation_path = Path(args.annotation_file).resolve()

    print("Extract annotations:")
    annotation_data = extract_annotations(annotation_path, args.limit)

    print("Generating file structure:")

    if args.img_dir == "./":
        img_dir = annotation_path.parent
    else:
        img_dir = (annotation_path.parent / args.img_dir).resolve()

    out = Path(args.out)
    partitioned_dataset = split_data(
        annotation_data, args.split, shuffle=not args.no_shuffle, seed=args.seed
    )

    for label, data in partitioned_dataset.items():
        if len(data) == 0:
            print(f"No {label} folder")
            continue

        generate_file_structure(
            data,
            classes,
            out / label,
            img_dir,
            padding=args.padding,
            copy_image=not args.no_image,
            clear_existing=args.clear_existing,
        )

    print("Generating yaml config")

    generate_config(out, partitioned_dataset, classes)

    print("Converted annotation into yolo format")

    if args.zip:
        shutil.make_archive("dataset", "zip", out / "..")
        print("Zipped dataset")
