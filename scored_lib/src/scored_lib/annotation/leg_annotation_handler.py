from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from time import time
from typing import Any, Iterator, NamedTuple

import cv2

import numpy as np
from scored_lib.annotation.throw_annotation import DartThrowAnnotation
from scored_lib.annotation.leg_info import LegAnnotation


class Sample(NamedTuple):
    """A single stored sample: its annotation and the linked camera frame."""

    annotation: DartThrowAnnotation
    image_path: Path | None


def _sort_key(directory: Path) -> tuple[int, int, str]:
    """Sort key ordering sample folders by round and then by throw.

    Parameters
    ----------
    directory : Path
        The sample folder to build a sort key for.

    Returns
    -------
    tuple[int, int, str]
        A key that sorts ``1_2`` before ``1_10`` and unknown folders last.
    """
    try:
        round_part, throw_part = directory.name.split("_", 1)
        return (int(round_part), int(throw_part), "")
    except ValueError:
        return (0, 0, directory.name)


@dataclass
class LegAnnotationHandler:
    """Handler for the annotations of a single leg on disk.

    A leg directory is laid out as::

        <leg>/
            leg.json          # general leg info, see write_leg_info
            1_1/              # one folder per sample, named "<round>_<throw>"
                image.png
                annotation.json
            1_2/
                image.png
                annotation.json

    Each ``annotation.json`` links to the camera frame sitting next to it, so a
    sample folder can be moved or copied without breaking the annotation.
    """

    directory: Path
    info: LegAnnotation

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.info.save(self.directory)

    @property
    def sample_count(self) -> int:
        """The number of samples currently stored in the leg directory."""
        return len(self._sample_directories())

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over all samples in round and throw order.

        Yields
        ------
        Sample
            The loaded annotation together with the path of the linked image.
            Samples unpack as ``annotation, image_path``.
        """
        for directory in self._sample_directories():
            round, throw = map(int, directory.name.split("_", 1))
            yield self.read(round, throw)

    def end_leg(self, is_won: bool) -> None:
        """Finalize the leg by updating the leg info and writing it to disk.

        Parameters
        ----------
        is_won : bool
            Whether the player won the leg
        """
        self.info.end(time(), is_won)
        self.info.save(self.directory)

    def sample_directory(self, round: int, throw: int, create: bool = False) -> Path:
        """Get the folder of a single sample.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        throw : int
            One-based index of the throw within the round.
        create : bool, optional
            Create the folder if it does not exist, by default False.

        Returns
        -------
        Path
            The path of the sample folder.
        """
        if round < 1:
            raise ValueError("round must be one-based and greater than zero")
        if throw < 1:
            raise ValueError("throw must be one-based and greater than zero")

        directory = self.directory / f"{round}_{throw}"
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def add(self, annotation: DartThrowAnnotation, image: np.ndarray) -> Sample:
        """Add a new sample for a throw.

        The sample folder is created, the image is written into it and the
        annotation is saved next to it, linked to that image.

        Parameters
        ----------
        annotation : DartThrowAnnotation
            The annotation to store, it carries the ``(round, throw)`` coordinates.
        image : np.ndarray
            The camera frame to store alongside the annotation. Accepts a numpy
            array representing the image.

        Returns
        -------
        Sample
            The stored sample.
        """
        directory = self.sample_directory(
            annotation.round, annotation.throw, create=True
        )

        if image is not None:
            cv2.imwrite(str(directory / f"image.jpg"), image)

        annotation.save(directory)

        return Sample(annotation, directory / "image.jpg")

    def read(self, round: int, throw: int) -> Sample:
        """Read the sample of a throw.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        throw : int
            One-based index of the throw within the round.

        Returns
        -------
        Sample
            The stored sample with its annotation and linked image path.
        """
        directory = self.sample_directory(round, throw)
        return Sample(DartThrowAnnotation.load(directory), directory / "image.jpg")

    def exists(self, round: int, throw: int) -> bool:
        """Check whether a sample exists for a throw.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        throw : int
            One-based index of the throw within the round.

        Returns
        -------
        bool
            True if a sample folder exists for the given throw, False otherwise.
        """
        return self.sample_directory(round, throw).exists()

    def edit(
        self,
        round: int,
        throw: int,
        annotation: DartThrowAnnotation | None = None,
        image: Any = None,
    ) -> Sample:
        """Edit an existing sample, either its annotation or its image.

        Passing only one of ``annotation`` or ``image`` keeps the other part of
        the sample as it is, including the image link of the annotation.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        throw : int
            One-based index of the throw within the round.
        annotation : DartThrowAnnotation | None, optional
            The replacement annotation, by default None. It has to be located
            at the same ``(round, throw)`` coordinates as the sample. When it
            carries no ``image_filename`` the existing image link is kept.
        image : Any, optional
            The replacement camera frame, by default None. Accepts a numpy array,
            the encoded bytes of an image, or a path to an image file. Passing
            an image replaces the linked one, the old file is removed.

        Returns
        -------
        Sample
            The updated sample.
        """
        if annotation is None and image is None:
            raise ValueError("Provide an annotation, an image, or both.")

        directory = self.sample_directory(round, throw)

        if annotation is not None:
            if annotation.index != (round, throw):
                raise ValueError(
                    f"Annotation is located at round {annotation.round} throw {annotation.throw} "
                    f"but is being edited as round {round} throw {throw}."
                )
            annotation.save(directory)
        if image is not None:
            cv2.imwrite(str(directory / f"image.jpg"), image)

        return Sample(
            annotation
            if annotation is not None
            else DartThrowAnnotation.load(directory),
            directory / "image.jpg",
        )

    def remove(self, round: int, throw: int) -> None:
        """Delete a sample and its annotation from disk.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        throw : int
            One-based index of the throw within the round.
        """
        shutil.rmtree(self.sample_directory(round, throw))

    def _sample_directories(self) -> list[Path]:
        """Collect the sample folders of the leg in round and throw order.

        Returns
        -------
        list[Path]
            The sample folders found in the leg directory.
        """
        directories = [
            entry
            for entry in self.directory.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        ]
        return sorted(directories, key=_sort_key)
