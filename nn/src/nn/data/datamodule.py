from pathlib import Path
from typing import NamedTuple

import pytorch_lightning as pl
import torch
from nn.data.config import DataModuleConfig
from nn.data.dataset import DartKeypointDataset, DatasetConfig
from nn.data.transform import Transform
from torch.utils.data import DataLoader


class Batch(NamedTuple):
    images: torch.Tensor
    gt_boxes: torch.Tensor
    gt_labels: torch.Tensor
    gt_keypoints: torch.Tensor
    annotation_paths: list[Path]


class DartKeyPointDataModule(pl.LightningDataModule):
    def __init__(
        self, config: DataModuleConfig, dataset: DatasetConfig, transforms: list[Transform]
    ):
        super().__init__()
        self.config = config
        self.transforms = transforms
        self.dataset = dataset

    def setup(self, stage: str):

        if stage == "fit":
            self.train_set = DartKeypointDataset(
                self.dataset.train,
                self.dataset.path,
                self.dataset.cls_dict,
                self.dataset.num_keypoints,
                [transform for transform in self.transforms if transform.train_only()],
            )
            self.val_set = DartKeypointDataset(
                self.dataset.val,
                self.dataset.path,
                self.dataset.cls_dict,
                self.dataset.num_keypoints,
                [transform for transform in self.transforms if not transform.train_only()],
            )
        elif stage == "test":
            self.test_set = DartKeypointDataset(
                self.dataset.test,
                self.dataset.path,
                self.dataset.cls_dict,
                self.dataset.num_keypoints,
                [transform for transform in self.transforms if not transform.train_only()],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Path]],
) -> Batch:
    """
    Custom collate function to handle batches of data.
    """
    images: list[torch.Tensor]
    gt_boxes: list[torch.Tensor]
    gt_labels: list[torch.Tensor]
    gt_keypoints: list[torch.Tensor]
    annotation_paths: list[Path]

    images, gt_boxes, gt_labels, gt_keypoints, annotation_paths = map(
        list, zip(*batch, strict=True)
    )

    image_batch = torch.stack(images, dim=0)
    gt_boxes_padded = torch.nn.utils.rnn.pad_sequence(gt_boxes, batch_first=True, padding_value=0.0)
    gt_labels_padded = torch.nn.utils.rnn.pad_sequence(
        gt_labels, batch_first=True, padding_value=-1
    )
    gt_keypoints_padded = torch.nn.utils.rnn.pad_sequence(
        gt_keypoints, batch_first=True, padding_value=0.0
    )

    return Batch(
        image_batch, gt_boxes_padded, gt_labels_padded, gt_keypoints_padded, annotation_paths
    )
