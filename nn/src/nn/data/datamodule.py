from dataclasses import dataclass

import pytorch_lightning as pl
from nn.data.dataset import DartKeypointDataset, DatasetConfig
from nn.data.transform import Transform
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DataModuleConfig:
    """
    Configuration for the datamodule
    """

    dataset_config: DatasetConfig
    batch_size: int
    num_workers: int
    prefetch_factor: int = 1
    pin_memory: bool = False


class DartKeyPointDataModule(pl.LightningDataModule):
    def __init__(self, config: DataModuleConfig, transforms: list[Transform]):
        super().__init__()
        self.config = config
        self.transforms = transforms

    def setup(self, stage: str):
        dataset = self.config.dataset_config

        if stage == "fit":
            self.train_set = DartKeypointDataset(
                dataset.train,
                [transform for transform in self.transforms if transform.train_only()],
            )
            self.val_set = DartKeypointDataset(
                dataset.val,
                [transform for transform in self.transforms if not transform.train_only()],
            )
        elif stage == "test":
            self.test_set = DartKeypointDataset(
                dataset.test,
                [transform for transform in self.transforms if not transform.train_only()],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
