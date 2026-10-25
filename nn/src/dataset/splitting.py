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
        return train, val, test


SPLITTERS = {
    "random": RandomSplitter,
}
