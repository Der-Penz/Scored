from pathlib import Path
from typing import NamedTuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nn.data.config import DataModuleConfig
from nn.data.dataset import DartKeypointDataset, DatasetConfig
from nn.data.transform import Resize, Transform


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
        resize = Resize(self.config.image_size)
        train_transforms = [resize, *[t for t in self.transforms if t.train_only()]]
        eval_transforms = [resize, *[t for t in self.transforms if not t.train_only()]]

        if stage == "fit":
            self.train_set = DartKeypointDataset(
                self.dataset.train,
                self.dataset.path,
                self.dataset.cls_dict,
                self.dataset.num_keypoints,
                train_transforms,
            )
            self.val_set = DartKeypointDataset(
                self.dataset.val,
                self.dataset.path,
                self.dataset.cls_dict,
                self.dataset.num_keypoints,
                eval_transforms,
            )
        elif stage == "test":
            self.test_set = DartKeypointDataset(
                self.dataset.test,
                self.dataset.path,
                self.dataset.cls_dict,
                self.dataset.num_keypoints,
                eval_transforms,
            )

    def _dataloader(self, dataset: DartKeypointDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            # PyTorch rejects a prefetch factor when no worker processes are used.
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_set, shuffle=False)


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
