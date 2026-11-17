import random
from pathlib import Path
from typing import Protocol


class Splitter(Protocol):
    def __init__(self, seed: int, val_ratio: float, test_ratio: float) -> None:
        """
        Initialize the splitter.

        Parameters
        ----------
        seed : int
            Seed of the random generator, so repeated runs produce the same split.
        val_ratio : float
            Fraction of samples placed in the validation split.
        test_ratio : float
            Fraction of samples placed in the test split.
        """

    def split_samples(self, samples: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        """
        Split the given samples into 3 splits

        Parameters
        ----------
        samples : list[Path]
            The full list of samples to split.

        Returns
        -------
        tuple[list[Path], list[Path], list[Path]]
            Train, validation and test samples.
        """
        ...


class RandomSplitter:
    def __init__(self, seed: int, val_ratio: float, test_ratio: float) -> None:
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random = random.Random(seed)

    def split_samples(self, samples: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        """
        Shuffle and split samples into train, validation and test sets.

        Parameters
        ----------
        samples : list[Path]
            All sample directories.

        Returns
        -------
        tuple[list[Path], list[Path], list[Path]]
            The train, validation and test samples.

        Raises
        ------
        ValueError
            If the ratios leave no samples for training.
        """
        shuffled = list(samples)
        self.random.shuffle(shuffled)

        num_test = int(len(shuffled) * self.test_ratio)
        num_val = int(len(shuffled) * self.val_ratio)

        test = shuffled[:num_test]
        val = shuffled[num_test : num_test + num_val]
        train = shuffled[num_test + num_val :]
        if not train:
            raise ValueError("The ratios leave no samples for training.")
        return train, val, test


class StratifiedSplitter:
    """
    Split whole legs into train, validation and test.

    Samples are grouped by their parent directory, so all throws of one leg end up in the
    same split and no leg leaks across splits. Groups are shuffled with the configured seed
    and handed to the split whose quota is filled the least, which keeps the split sizes
    close to the configured ratios even when legs contain a different number of throws.

    Parameters
    ----------
    seed : int
        Seed of the random generator, so repeated runs produce the same split.
    val_ratio : float
        Fraction of samples placed in the validation split.
    test_ratio : float
        Fraction of samples placed in the test split.
    """

    def __init__(self, seed: int, val_ratio: float, test_ratio: float) -> None:
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random = random.Random(seed)

    def split_samples(self, samples: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        """
        Split samples along their leg directories into 3 splits.

        Parameters
        ----------
        samples : list[Path]
            All sample directories.

        Returns
        -------
        tuple[list[Path], list[Path], list[Path]]
            The train, validation and test samples.

        Raises
        ------
        ValueError
            If the ratios leave no samples for training.
        """
        groups: dict[Path, list[Path]] = {}
        for sample in samples:
            groups.setdefault(sample.parent, []).append(sample)

        shuffled_groups = list(groups.values())
        self.random.shuffle(shuffled_groups)

        total = len(samples)
        targets = {
            "train": total - int(total * self.val_ratio) - int(total * self.test_ratio),
            "val": int(total * self.val_ratio),
            "test": int(total * self.test_ratio),
        }
        splits: dict[str, list[Path]] = {name: [] for name in targets}

        for group in shuffled_groups:
            name = min(
                (name for name, target in targets.items() if target > 0),
                key=lambda split: len(splits[split]) / targets[split],
                default="train",
            )
            splits[name].extend(group)

        train, val, test = splits["train"], splits["val"], splits["test"]
        if not train:
            raise ValueError("The ratios leave no samples for training.")
        return train, val, test


SPLITTERS = {
    "random": RandomSplitter,
    "stratified": StratifiedSplitter,
}
