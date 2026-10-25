import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
import torchvision.transforms as T
from nn.data.transform import Transform
from PIL import Image
from scored_lib.annotation.image_annotation import ImageAnnotation
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """
    Base class for datasets.
    """

    name: str
    description: str
    num_keypoints: int
    cls_dict: dict[str, int]
    path: Path
    train: list[str]
    val: list[str]
    test: list[str]


class SampleIndex(TypedDict):
    image_path: Path
    annotation: ImageAnnotation
    annotation_path: Path


class DartKeypointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[str],
        path: Path,
        cls_dict: dict[str, int],
        num_keypoints: int,
        transforms: list[Transform],
    ):
        super().__init__()
        self.samples = samples
        self.path = path
        self.cls_dict = cls_dict
        self.num_keypoints = num_keypoints
        self.transforms = transforms
        self.index: list[SampleIndex] = self.build_index()

    def build_index(self) -> list[SampleIndex]:
        """
        Build an index for the dataset.
        """

        index: list[SampleIndex] = []

        for sample in tqdm(self.samples, desc="Building dataset index", unit="sample"):
            sample_path = self.path / sample

            image_path = self.get_image_path(sample_path)
            annotation_path = sample_path / "annotation.json"

            if not annotation_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

            try:
                annotation = self.load_annotation(annotation_path)
            except FileNotFoundError as err:
                raise FileNotFoundError(f"Annotation file not found: {annotation_path}") from err
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Failed to decode JSON from annotation file: {annotation_path}"
                ) from err

            index.append(
                {
                    "image_path": image_path,
                    "annotation": annotation,
                    "annotation_path": annotation_path,
                }
            )

        return index

    def get_image_path(self, sample_path: Path) -> Path:
        img_extensions = [".png", ".jpg", ".jpeg"]
        for ext in img_extensions:
            image_path = sample_path / f"image{ext}"
            if image_path.exists():
                return image_path
        raise FileNotFoundError(
            f"No image file found in {sample_path} with extensions {img_extensions}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample: SampleIndex = self.index[idx]
        except IndexError as err:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self)}"
            ) from err

        try:
            image = self.load_image(sample["image_path"])
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Image file not found: {sample['image_path']}") from err

        img_size: tuple[int, int] = image.shape[1:]  # type: ignore # (C, H, W) -> (H, W)
        gt_boxes, gt_labels, gt_keypoints = self.extract_annotation_data(
            sample["annotation"], img_size
        )

        for transform in self.transforms:
            image, gt_boxes, gt_labels, gt_keypoints = transform(
                image, gt_boxes, gt_labels, gt_keypoints
            )

        return image, gt_boxes, gt_labels, gt_keypoints, sample["annotation_path"]

    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load an image from the given path and convert it to a torch.Tensor.
        """

        image = Image.open(image_path).convert("RGB")
        transform = T.ToTensor()
        return transform(image)

    def load_annotation(self, annotation_path: Path) -> ImageAnnotation:
        """
        Load an annotation from the given path and return it as an ImageAnnotation object.
        """
        return ImageAnnotation.from_json(annotation_path.read_text())

    def extract_annotation_data(
        self, annotation: ImageAnnotation, img_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract ground truth keypoints and build the bbox from the keypoints.
        """

        h, w = img_size

        gt_boxes: list[list[float]] = []
        gt_labels: list[int] = []
        gt_keypoints: list[torch.Tensor] = []

        for annotated_throw in annotation.throws:
            keypoints = annotated_throw.keypoints

            if keypoints is None:
                print(
                    f"No keypoints found for throw {annotated_throw.throw_data.label} in annotation"
                    f"{annotation.round}-{annotation.throw}. Skipping this throw."
                )
                continue

            gt_labels.append(self.cls_dict["dart"])

            keypoints_tensor = torch.zeros((self.num_keypoints, 3), dtype=torch.float32)

            actual_kpts = torch.tensor(
                [
                    [*keypoints.tip, float(keypoints.tip_visible)],
                    [*keypoints.flight, float(keypoints.flight_visible)],
                ],
                dtype=torch.float32,
            )
            num_actual = actual_kpts.shape[0]
            keypoints_tensor[:num_actual] = actual_kpts

            gt_keypoints.append(keypoints_tensor)

            # compute bounding box from keypoints with padding from the tip to the flight
            tip, flight = keypoints.tip, keypoints.flight
            padding_h, padding_w = 2 / h, 2 / w  # Add some padding to the bounding box
            x_min = max(min(tip[0], flight[0]) - padding_w, 0)
            y_min = max(min(tip[1], flight[1]) - padding_h, 0)
            x_max = min(max(tip[0], flight[0]) + padding_w, 1)
            y_max = min(max(tip[1], flight[1]) + padding_h, 1)

            gt_boxes.append([x_min, y_min, x_max, y_max])

        if len(gt_boxes) == 0:
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, self.num_keypoints, 3), dtype=torch.float32),
            )

        return (
            torch.tensor(gt_boxes, dtype=torch.float32),
            torch.tensor(gt_labels, dtype=torch.long),
            torch.stack(gt_keypoints, dim=0),
        )
