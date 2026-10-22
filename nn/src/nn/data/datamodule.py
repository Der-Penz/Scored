import pytorch_lightning as pl
from nn.data.config import DataModuleConfig
from nn.data.dataset import DartKeypointDataset, DatasetConfig
from nn.data.transform import Transform
from torch.utils.data import DataLoader


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
                [transform for transform in self.transforms if transform.train_only()],
            )
            self.val_set = DartKeypointDataset(
                self.dataset.val,
                [transform for transform in self.transforms if not transform.train_only()],
            )
        elif stage == "test":
            self.test_set = DartKeypointDataset(
                self.dataset.test,
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
