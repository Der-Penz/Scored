from math import ceil
from typing import Any, Literal, Sequence, Tuple, Dict
import numpy as np


def split_data(
    data: Sequence[Any],
    split_ratios: Tuple[float, float, float],
    shuffle=True,
    seed: int = None,
) -> Dict[Literal["train", "val", "test"], Sequence[Any]]:
    """
    Splits the data into train, validation and test sets.

    :param data: The data to split
    :param split_ratios: The ratios to split the data into. The first element is the ratio for the training set, the second for the validation set and the third for the test set.
    :param shuffle: Whether to shuffle the data before splitting
    :param seed: The seed for the random number generator
    :return: A dictionary containing the splitted data with the keys "train", "val" and "test"
    """
    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(data)

    train_size = ceil(len(data) * split_ratios[0])
    val_size = ceil(len(data) * split_ratios[1])

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]
    return {"train": train_data, "val": val_data, "test": test_data}
