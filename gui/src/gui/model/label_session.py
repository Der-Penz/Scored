"""Session state for labeling a folder of images outside a game."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scored_lib.annotation.image_annotation import ImageAnnotation
from scored_lib.annotation.sample_store import (
    Sample,
    discover_samples,
    write_image_annotation,
)


class LabelSession:
    """The images of one folder and which of them is being labeled.

    A session holds paths and annotations only; the frame of the image on
    screen is loaded on demand, so a folder of thousands of photos costs a list
    of paths and not a list of images.

    Parameters
    ----------
    read_only : bool
        True to inspect the labels without ever writing them back to disk.
    """

    def __init__(self, read_only: bool = False) -> None:
        self.read_only = read_only
        self.directory: Path | None = None
        self.samples: list[Sample] = []
        self.index = 0
        self.dirty = False

    @property
    def current(self) -> Sample | None:
        """The sample on screen, or None while the session is empty."""
        if not 0 <= self.index < len(self.samples):
            return None
        return self.samples[self.index]

    @property
    def annotation(self) -> ImageAnnotation | None:
        """The annotation of the sample on screen."""
        sample = self.current
        return sample.annotation if sample is not None else None

    @property
    def image_path(self) -> Path | None:
        """The path of the image on screen."""
        sample = self.current
        return sample.image_path if sample is not None else None

    @property
    def position(self) -> str:
        """The one-based position of the sample on screen, e.g. ``12 / 348``."""
        if not self.samples:
            return "0 / 0"
        return f"{self.index + 1} / {len(self.samples)}"

    def open(self, directory: Path) -> int:
        """Load the samples below *directory* and show the first one.

        Parameters
        ----------
        directory : Path
            The folder to scan for images and existing annotations.

        Returns
        -------
        int
            The number of samples that were found.
        """
        self.directory = directory
        try:
            self.samples = discover_samples(directory)
        except OSError as error:
            logging.error(f"Could not read the image folder {directory}: {error}")
            self.samples = []
        self.index = 0
        self.dirty = False
        logging.info(f"Labeling {len(self.samples)} samples from {directory}")
        return len(self.samples)

    def image(self) -> np.ndarray | None:
        """Load the frame of the sample on screen as an RGB array."""
        path = self.image_path
        if path is None:
            return None
        try:
            return np.asarray(Image.open(path).convert("RGB"))
        except (OSError, ValueError) as error:
            logging.error(f"Could not read the image {path}: {error}")
            return None

    def step(self, offset: int) -> bool:
        """Move to another sample without wrapping around.

        Parameters
        ----------
        offset : int
            How many samples to move, negative to go back.

        Returns
        -------
        bool
            True if the session moved, False at the first or last sample.
        """
        target = self.index + offset
        if not 0 <= target < len(self.samples):
            return False
        self.index = target
        self.dirty = False
        return True

    def save(self) -> Path | None:
        """Write the annotation of the sample on screen next to its image.

        Returns
        -------
        Path | None
            The label file that was written, or None when there was nothing to
            save, the session is read only, or writing failed.
        """
        annotation = self.annotation
        path = self.image_path
        if annotation is None or path is None or self.read_only:
            return None
        try:
            target = write_image_annotation(path, annotation)
        except OSError as error:
            logging.error(f"Could not save the labels for {path}: {error}")
            return None
        self.dirty = False
        logging.debug(f"Wrote {target}")
        return target
