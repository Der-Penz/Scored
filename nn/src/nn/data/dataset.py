import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
import torchvision.transforms as T
from nn.data.transform import Transform
from PIL import Image
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
    annotation: dict
    annotation_path: Path


class DartKeypointDataset(torch.utils.data.Dataset):
    def __init__(
        self, samples: list[str], path: Path, cls_dict: dict[str, int], transforms: list[Transform]
    ):
        super().__init__()
        self.samples = samples
        self.path = path
        self.cls_dict = cls_dict
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

            annotation = self.load_annotation(annotation_path)

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

    def load_annotation(self, annotation_path: Path) -> dict:
        return json.loads(annotation_path.read_text())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample: SampleIndex = self.index[idx]
        except IndexError as err:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self)}"
            ) from err

        image = self.load_image(sample["image_path"])
        gt_boxes = torch.tensor(sample["annotation"]["boxes"], dtype=torch.float32)
        gt_labels = torch.tensor(sample["annotation"]["labels"], dtype=torch.int64)
        gt_keypoints = torch.tensor(sample["annotation"]["keypoints"], dtype=torch.float32)

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
