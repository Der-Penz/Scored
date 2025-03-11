import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
import argparse
import shutil


def get_data_from_json(json_data):
    return np.array([json_data["x"] / 100, json_data["y"] / 100]), (json_data["original_width"], json_data["original_height"]), json_data["keypointlabels"][0]

def compute_bounding_box(key_points : np.ndarray, padding : int = 0):

    x, y = np.stack(key_points).T
    min_x, min_y = max(np.min(x) - padding, 0), max(np.min(y) - padding, 0)
    max_x, max_y = min(np.max(x) + padding, 1), min(np.max(y) + padding, 1)

    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    width, height = max_x - min_x, max_y - min_y
    return (center, width, height)

def to_absolute_pixels(key_point, shape):
    return np.array([key_point[0] * shape[0], key_point[1] * shape[1]])

def transform_annotation(annotation_data):
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
    return keypoints, labels

def extract_annotation(annotations_path : str, keypoint_column = "kp-1"):
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
            transformed.append((img_name, ) + transform_annotation(parsed))
        except Exception as e:
            print(f"Error parsing entry {i + 1}")
            print(entry)
            print(e)

    loading_bar(total, total)
    print()    

    return transformed

def generate_file_structure(annotation_data, out_path, img_dir, keypoint_order, copy_image=True, **kwargs):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Directory {out_path} created.")
    else:
        print(f"Directory {out_path} already exists.")

    total = len(annotation_data)

    out = Path(out_path)
    for i, (img_name, keypoints, labels) in enumerate(annotation_data):
        loading_bar(i, total)
        img_path = img_dir / img_name
        if not img_path.exists():
            print("missing image file" + str(img_path))

        img_out = out / "images"
        if not img_out.exists():
            os.makedirs(img_out)
        img_out = img_out / img_name

        if copy_image:
            shutil.copy(img_path, img_out)

        annotation_file = out / "labels"
        if not annotation_file.exists():
            os.makedirs(annotation_file)
        annotation_file = annotation_file / (img_name.split(".")[0] + ".txt")
        with open(annotation_file, "w") as file:
            for order in keypoint_order:
                try:
                    annotation = get_annotation_in_yolo_format(keypoints, labels, order, **kwargs)
                except Exception as e:
                    print(f"\nError: {str(e)} for image {img_name}")
                file.write(" ".join(map(lambda x : str(x), annotation)))
                file.write("\n")

    loading_bar(total, total)
    print()    

def get_annotation_in_yolo_format(keypoints, labels, keypoint_order, **kwargs):
    class_index = 0

    keypoint_dict = dict(zip(map(lambda x : x.lower(), labels), keypoints))
    try:
        keypoints = [keypoint_dict[label.lower()] for label in keypoint_order]
    except KeyError as e:
        raise Exception(f"Missing keypoint: {e}. Needed keypoints: {keypoint_order}, available keys: {labels}") from e
    bb = compute_bounding_box(keypoints, **kwargs)
    bb_center_x = bb[0][0]
    bb_center_y = bb[0][1]
    bb_width = bb[1]
    bb_height = bb[2]

    line = [class_index, bb_center_x, bb_center_y, bb_width, bb_height]

    for keypoint in keypoints:
        line.append(keypoint[0])
        line.append(keypoint[1])
    return line

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

def split_data(data, split_ratios):
    np.random.shuffle(data)
    train_size = int(len(data) * split_ratios[0])
    val_size = int(len(data) * split_ratios[1])
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    return (train_data, val_data, test_data)

def generate_config(out, labels, splitted_data, classes, keypoint_order):
    config = []

    config.append(f"path: {str(out)}")

    for label, data in zip(labels, splitted_data):
        if len(data) == 0:
            config.append(f"{label}: #empty")
        else:
            config.append(f"{label}: {label}/images")

    config.append("")
    config.append(f"nc: {len(classes)}")
    config.append(f"names:")
    for i, class_name in enumerate(classes):
        config.append(f"  {i}: {class_name}")
    config.append("")

    max_keypoints = len(max(keypoint_order, key=lambda x : len(x)))
    config.append(f"nkpt: {max_keypoints}")
    config.append(f"kpt_shape: [{max_keypoints}, 3]")
    config.append(f"keypoint_names:")
    for class_name, keypoints in zip(classes, keypoint_order): 
        config.append(f"  {class_name}: {str(keypoints)}")

    with open(out / "dataset.yaml", "w") as file:
        for line in config:
            file.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(  
        description='convert label studio csv data annotation format into yolo format')  
    parser.add_argument(
        "annotation-file", type=is_valid_csv,
        help="The CSV annotation file"
    )
    parser.add_argument(
        "--img-dir", type=is_valid_directory, default="./",
        help="The location of the image files relative to the annotation file (default: same directory)"
    )
    parser.add_argument(
        "--out", type=str, default="./train/data",
        help="The output directory where the files are saved"
    )
    parser.add_argument(
        "--padding", type=int, default=0,
        help="Padding in percentage of the img size around the bounding box"
    )
    parser.add_argument(
        "--keypoint-column", type=str, default="kp-1",
        help="The column name for the keypoint data"
    )
    parser.add_argument(
        "--keypoint-order", type=str, nargs="*", action="append", required=True,
        help="The keypoints in the order they will appear in the generated files"
    )

    parser.add_argument(
        "--classes", type=str, nargs="*",
        help="The classes for the different keypoints, used in combination if multiple keypoint_order are provided"
    )

    parser.add_argument(
        "--split", type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'), default=[0.7, 0.0, 0.3],
        help="Proportions for splitting the dataset into train, val, and test sets. Must sum to 1.0"
    )

    parser.add_argument(
        "--no-image", action="store_true",
        help="Do not copy the images into the output dataset"
    )

    parser.add_argument(
        "--zip", action="store_true",
        help="zip the resulting dataset"
    )
    
    args = parser.parse_args()  

    if sum(args.split) != 1:
        parser.error("--split must add up to 1.")

    if args.classes and not args.keypoint_order:
        parser.error("--classes can only be used if --keypoint_order is provided.")

    if args.keypoint_order and args.classes:
        if len(args.keypoint_order) != len(args.classes) and len(args.classes) != 0:
            parser.error("The number of --classes must match the number of --keypoint_order provided.")
    else:
        args.classes = list(range(len(args.keypoint_order)))

    print("Extract annotations:")
    annotation_data = extract_annotation(args.annotation_file, args.keypoint_column)

    print("Generating txt files")

    annotation_path = Path(args.annotation_file).resolve()
    if args.img_dir == "./":
        img_dir = annotation_path.parent
    else:
        img_dir = (annotation_path.parent / args.img_dir).resolve()
    
    out = Path(args.out)

    labels = ["train", "val", "test"]
    splitted_data = split_data(annotation_data, args.split)

    for label, data in zip(labels, splitted_data):
        if len(data) == 0:
            print(f"No {label} folder")
            continue

        print(f"Generating {label} folder")
        generate_file_structure(data, out / label, img_dir, args.keypoint_order, padding=args.padding, copy_image=not args.no_image)
    
    print("Generating yaml config")
        
    generate_config(out, labels, splitted_data, args.classes, args.keypoint_order)

    print("Converted annotation into yolo format")

    if args.zip:
        shutil.make_archive("dataset", 'zip', out)
        print("Zipped dataset")
    