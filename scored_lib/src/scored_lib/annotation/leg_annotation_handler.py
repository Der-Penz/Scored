from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import NamedTuple

import cv2
import numpy as np

from scored_lib.annotation.image_annotation import ImageAnnotation
from scored_lib.annotation.leg_annotation import LegAnnotation

LEG_JSON = "leg.json"
ANNOTATION_JSON = "annotation.json"
IMAGE_FILE = "image.jpg"

DELETED_MARKER = "d"
DELETED_SUFFIX = f"_{DELETED_MARKER}"

_SAMPLE_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)$")
_REMOVED_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)(?:_d+)$")


class Sample(NamedTuple):
    """A single stored sample: its annotation and the linked camera frame."""

    annotation: ImageAnnotation
    image_path: Path


def _parse_sample_dir(name: str) -> tuple[int, int] | None:
    """Parse a sample folder name into its one-based (round, throw) index.

    Parameters
    ----------
    name : str
        The folder name to parse.

    Returns
    -------
    tuple[int, int] | None
        The one-based round and throw index, or None if *name* is not a
        canonical sample folder, e.g. because it was removed or is unrelated.
    """
    match = _SAMPLE_DIR_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _is_removed_dir(name: str) -> bool:
    """Check whether a folder name belongs to a removed sample.

    Parameters
    ----------
    name : str
        The folder name to check.

    Returns
    -------
    bool
        True if the folder is a tombstone of a removed sample, False otherwise.
    """
    return _REMOVED_DIR_PATTERN.match(name) is not None


def _sample_sort_key(directory: Path) -> tuple[int, int]:
    """Sort key ordering sample folders by round and then by throw.

    Parameters
    ----------
    directory : Path
        The sample folder to build a sort key for.

    Returns
    -------
    tuple[int, int]
        The one-based round and throw index, so that ``1_2`` sorts before ``1_10``.
    """
    return _parse_sample_dir(directory.name) or (0, 0)


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

    A removed sample is either deleted (:meth:`remove`) or kept as a tombstone
    with a ``_d`` suffix (``1_1_d``), see :meth:`mark_removed`. Tombstoned
    folders are ignored by :attr:`sample_count` and :meth:`__iter__`.
    """

    directory: Path
    info: LegAnnotation

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._save_leg_info()

    @property
    def sample_count(self) -> int:
        """The number of samples currently stored in the leg directory."""
        return len(self._sample_directories())

    @property
    def deleted_count(self) -> int:
        """The number of removed samples kept as tombstones in the leg directory."""
        return sum(
            1
            for entry in self.directory.iterdir()
            if entry.is_dir() and _is_removed_dir(entry.name)
        )

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over all samples in round and throw order.

        Removed samples are skipped, they no longer describe a registered throw.

        Yields
        ------
        Sample
            The loaded annotation together with the path of the linked image.
            Samples unpack as ``annotation, image_path``.
        """
        for directory in self._sample_directories():
            index = _parse_sample_dir(directory.name)
            if index is None:  # unreachable, filtered out above
                continue
            yield self.read(*index)

    def _save_leg_info(self) -> None:
        """Write the leg info to disk."""
        self.directory.joinpath(LEG_JSON).write_text(
            self.info.to_json(), encoding="utf-8"
        )

    def end_leg(self, is_won: bool) -> None:
        """Finalize the leg by updating the leg info and writing it to disk.

        Parameters
        ----------
        is_won : bool
            Whether the player won the leg
        """
        self.info.end(time(), is_won)
        self._save_leg_info()

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

    def save(self, annotation: ImageAnnotation, image: np.ndarray) -> Sample:
        """Save a new ImageAnnotation

        Parameters
        ----------
        annotation : ImageAnnotation
            The annotation to store.
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
            cv2.imwrite(
                str(directory / IMAGE_FILE), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

        directory.joinpath(ANNOTATION_JSON).write_text(
            annotation.to_json(), encoding="utf-8"
        )

        logging.debug(
            f"Stored sample {annotation.round}_{annotation.throw} in {self.directory}"
        )
        return Sample(annotation, directory / IMAGE_FILE)

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
        return Sample(
            ImageAnnotation.from_json(directory.read_text()), directory / IMAGE_FILE
        )

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

    def remove(self, round: int, throw: int) -> None:
        """
        Delete a sample and its annotation from disk.

        Does nothing if no sample exists for the given throw.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        throw : int
            One-based index of the throw within the round.
        """
        directory = self.sample_directory(round, throw)
        if not directory.exists():
            return
        shutil.rmtree(directory)

    def mark_removed(self, round: int, throw: int) -> Path | None:
        """
        Keep a removed sample on disk as a tombstone instead of deleting it.

        The folder is renamed to ``<round>_<throw>_d``, appending another ``d``
        while a folder of that name is already taken. A throw that is recorded,
        removed and recorded again therefore keeps all of its frames: ``1_1``,
        then ``1_1_d``, then ``1_1_dd``.

        The stored frame is renamed to ``image.removed`` so the tombstone is not
        picked up by the Label Studio local storage image filter.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        throw : int
            One-based index of the throw within the round.

        Returns
        -------
        Path | None
            The tombstone folder, or None if there was no sample to mark.
        """
        directory = self.sample_directory(round, throw)
        if not directory.exists():
            return None

        count = 1
        while True:
            target = directory.with_name(f"{directory.name}_{DELETED_MARKER * count}")
            if not target.exists():
                break
            count += 1

        directory.rename(target)

        logging.debug(f"Marked sample {round}_{throw} as removed in {target}")
        return target

    def save_round(
        self, round: int, annotations: list[ImageAnnotation], images: list[np.ndarray]
    ) -> None:
        """Save all throws of a round.

        Parameters
        ----------
        round : int
            One-based index of the round within the leg.
        annotations : list[ImageAnnotation]
            The annotations to store for the throws of the round.
        images : list[np.ndarray]
            The images to store for the throws of the round.
        """
        for i, (annotation, image) in enumerate(zip(annotations, images)):
            if annotation.round != round:
                raise ValueError(
                    f"annotation round {annotation.round} does not match {round}"
                )
            if annotation.throw != i + 1:
                raise ValueError(
                    f"annotation throw {annotation.throw} does not match {i + 1}"
                )

            self.save(annotation, image)

    def _sample_directories(self) -> list[Path]:
        """Collect the live sample folders of the leg in round and throw order.

        Returns
        -------
        list[Path]
            The canonical ``<round>_<throw>`` folders found in the leg
            directory, ignoring removed samples and unrelated folders.
        """
        directories = [
            entry
            for entry in self.directory.iterdir()
            if entry.is_dir() and _parse_sample_dir(entry.name) is not None
        ]
        return sorted(directories, key=_sample_sort_key)
