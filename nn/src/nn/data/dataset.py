from dataclasses import dataclass

import torch
from nn.data.transform import Transform


@dataclass
class DatasetConfig:
    """
    Base class for datasets.
    """

    name: str
    description: str
    num_keypoints: int
    num_classes: int
    train: dict[str, dict]
    val: dict[str, dict]
    test: dict[str, dict]


class DartKeypointDataset(torch.utils.data.Dataset):
    def __init__(self, db: dict[str, dict], transforms: list[Transform]):
        self.db = [(key, value) for key, value in db.items()]
        self.transforms = transforms

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        sample = self.get_sample(idx)

        for transform in self.transforms:
            sample = transform(sample)

        return sample, idx

    def get_sample(self, idx):
        return self.db[idx][1]
