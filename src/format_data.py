import os
from pathlib import Path
import time
from typing import Tuple
import pandas as pd
import numpy as np
import json
import sys
import argparse
import shutil


def get_data_from_json(json_data):
    return np.array([json_data["x"] / 100, json_data["y"] / 100]), (json_data["original_width"], json_data["original_height"]), json_data["keypointlabels"][0]

def compute_bounding_box(key_points : np.ndarray, shape : Tuple[int, int], padding : int = 0):
    rel_padding_x, rel_padding_y = padding / shape[0], padding / shape[1]

    x, y = np.stack(key_points).T
    min_x, min_y = max(np.min(x) - rel_padding_x, 0), max(np.min(y) - rel_padding_y, 0)
    max_x, max_y = min(np.max(x) + rel_padding_x, shape[0]), min(np.max(y) + rel_padding_y, shape[1])

    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    width, height = max_x - min_x, max_y - min_y
    return (center, width, height)

def to_absolute_pixels(key_point, shape):
    return np.array([key_point[0] * shape[0], key_point[1] * shape[1]])

def transform_annotation(annotation_data, **kwargs):
    keypoints = []
    labels = []
    shape = None
    for point in annotation_data:
        position, s, keypoint_label = get_data_from_json(point)
        if shape is None:
            shape = s
        else:
            assert shape == s, "Shapes for different keypoints in the same image must match" 

        keypoints.append(position)
        labels.append(keypoint_label)
    bb = compute_bounding_box(keypoints, shape, **kwargs)
    return keypoints, labels, bb

def extract_annotation(annotations_path : str, keypoint_column = "kp-1", **kwargs):
    df = pd.read_csv(annotations_path)

    total = df.shape[0]
    transformed = []
    for i, (entry, img) in enumerate(df[[keypoint_column, "img"]].itertuples(index=False, name=None)):
        if entry is None:
            continue

        if "dartboard" in img:
            img_name = "dartboard" + img.split("dartboard", 1)[-1]
        else:
            print("Img names must contain the string dartboard")
            continue
        try:
            loading_bar(i, total)
            parsed = json.loads(entry)
            transformed.append((img_name, ) + transform_annotation(parsed, **kwargs))
        except Exception as e:
            print(f"Error parsing entry {i + 1}")
            print(entry)
            print(e)

    loading_bar(total, total)
    print()    

    return transformed

def generate_file_structure(annotation_data, out_path, img_dir, keypoint_order):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Directory {out_path} created.")
    else:
        print(f"Directory {out_path} already exists.")

    total = len(annotation_data)

    out = Path(out_path)
    for i, (img_name, keypoints, labels, bb) in enumerate(annotation_data):
        loading_bar(i, total)
        img_path = img_dir / img_name
        if not img_path.exists():
            print("missing image file" + str(img_path))

        img_out = out / img_name
        shutil.copy(img_path, img_out)

        annotation_file = out / (img_name.split(".")[0] + ".txt")
        with open(annotation_file, "w") as file:
            class_index = 0
            bb_center_x = bb[0][0]
            bb_center_y = bb[0][1]
            bb_width = bb[1]
            bb_height = bb[2]
            keypoint_dict = dict(zip(map(lambda x : x.lower(), labels), keypoints))
            keypoints = [keypoint_dict[label] for label in keypoint_order]

            line = [class_index, bb_center_x, bb_center_y, bb_width, bb_height]

            for keypoint in keypoints:
                line.append(keypoint[0])
                line.append(keypoint[1])
            file.write(" ".join(map(lambda x : str(x), line)))

    loading_bar(total, total)
    print()    


def is_valid_directory(path):
    """Check if the given path is a valid directory."""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Invalid directory: {path}")
    return path

def is_valid_csv(path):
    """Check if the given path is a valid CSV file."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")

    if not path.lower().endswith(".csv"):
        raise argparse.ArgumentTypeError(f"Invalid file type: {path} (expected a .csv file)")
    
    return path


def loading_bar(iteration, total, length=30):
    """Prints a CLI progress bar."""
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    
    sys.stdout.write(f"\r[{bar}] {percent:.1f}%")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(  
        description='convert label studio csv data annotation format into yolo format')  
    parser.add_argument(
        "annotation_file", type=is_valid_csv,
        help="The CSV annotation file"
    )
    parser.add_argument(
        "--img_dir", type=is_valid_directory, default="./",
        help="The location of the image files relative to the annotation file (default: same directory)"
    )
    parser.add_argument(
        "--out", type=str, default="./out",
        help="The output directory where the files are saved"
    )
    parser.add_argument(
        "--padding", type=int, default=0,
        help="Padding in pixels around the bounding box"
    )
    parser.add_argument(
        "--keypoint_column", type=str, default="kp-1",
        help="The column name for the keypoint data"
    )
    parser.add_argument(
        "--keypoint_order", type=str, nargs="*", default="top left bottom right",
        help="The keypoints in the order they will appear in the generated files"
    )
    args = parser.parse_args()  

    print("Extract annotations:")
    annotation_data = extract_annotation(args.annotation_file, args.keypoint_column, padding=args.padding)

    print("Generating txt files")

    annotation_path = Path(args.annotation_file).resolve()
    if args.img_dir == "./":
        img_dir = annotation_path.parent
    else:
        img_dir = (annotation_path.parent / args.img_dir).resolve()
    
    generate_file_structure(annotation_data, args.out, img_dir, args.keypoint_order.split(" "))

    print("Converted annotation into yolo format")
    