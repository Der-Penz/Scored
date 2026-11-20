"""Read, write and discover annotation samples on disk.

A sample is one camera frame together with the :class:`ImageAnnotation` that
labels the darts on it. A recorded leg keeps one folder per sample::

    <leg>/<round>_<throw>/
        image.jpg
        annotation.json

A folder of existing photos cannot share one ``annotation.json``, so those
images keep their name and get a JSON file next to them named after the image
(``IMG_1234.jpg`` next to ``IMG_1234.json``). Both layouts use the same
annotation format and are read by :func:`discover_samples`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from scored_lib.annotation.image_annotation import ImageAnnotation

LEG_JSON = "leg.json"
ANNOTATION_JSON = "annotation.json"
IMAGE_FILE = "image.jpg"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

_SAMPLE_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)$")
_REMOVED_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)(?:_d+)$")


class Sample(NamedTuple):
    """A single stored sample: its annotation and the linked camera frame."""

    annotation: ImageAnnotation
    image_path: Path


def parse_sample_dir(name: str) -> tuple[int, int] | None:
    """Parse a sample folder name into its one-based ``(round, throw)`` index.

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


def is_removed_sample(name: str) -> bool:
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


def natural_key(path: Path | str) -> tuple:
    """Build a sort key that orders ``IMG_2`` before ``IMG_10``.

    Parameters
    ----------
    path : Path | str
        The path to build a sort key for.

    Returns
    -------
    tuple
        A comparable key with the digit runs split off as their own elements.
    """
    parts = re.split(r"(\d+)", str(path))
    return tuple((1, int(part)) if part.isdigit() else (0, part.lower()) for part in parts)


def annotation_path_for(image_path: Path) -> Path:
    """Return the file the labels of *image_path* belong in.

    A recorded frame lives in a folder of its own and keeps the shared
    ``annotation.json`` name. Any other image gets a JSON file named after it,
    so several photos can share a folder.

    Parameters
    ----------
    image_path : Path
        The image whose label file is wanted.

    Returns
    -------
    Path
        The path of the label file, whether it exists or not.
    """
    if image_path.name == IMAGE_FILE:
        return image_path.parent / ANNOTATION_JSON
    return image_path.with_suffix(".json")


def image_path_of(directory: Path) -> Path:
    """Return the image stored in a sample *directory*.

    Parameters
    ----------
    directory : Path
        The sample folder to look in.

    Returns
    -------
    Path
        The path of the image, falling back to :data:`IMAGE_FILE` when the
        folder holds no recognizable image.
    """
    image = directory / IMAGE_FILE
    if image.is_file():
        return image
    candidates = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not candidates:
        return image
    return min(candidates, key=natural_key)


def read_sample(directory: Path) -> Sample:
    """Read the sample stored in a folder of its own.

    Parameters
    ----------
    directory : Path
        The folder holding ``annotation.json`` and the image.

    Returns
    -------
    Sample
        The stored annotation together with the linked image path.
    """
    annotation = ImageAnnotation.from_json(
        (directory / ANNOTATION_JSON).read_text(encoding="utf-8")
    )
    return Sample(annotation, image_path_of(directory))


def write_sample(
    directory: Path, annotation: ImageAnnotation, image: np.ndarray | None = None
) -> Sample:
    """Store *annotation* in *directory*, optionally next to *image*.

    Parameters
    ----------
    directory : Path
        The folder to write the sample to, created if it is missing.
    annotation : ImageAnnotation
        The annotation to store.
    image : np.ndarray | None, optional
        An RGB frame to store as ``image.jpg``, by default None to leave an
        already existing image untouched.

    Returns
    -------
    Sample
        The stored annotation together with the linked image path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if image is not None:
        cv2.imwrite(str(directory / IMAGE_FILE), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (directory / ANNOTATION_JSON).write_text(annotation.to_json(), encoding="utf-8")
    return Sample(annotation, image_path_of(directory))


def write_image_annotation(image_path: Path, annotation: ImageAnnotation) -> Path:
    """Store the labels of an existing image next to it.

    Parameters
    ----------
    image_path : Path
        The image the labels belong to.
    annotation : ImageAnnotation
        The annotation to store.

    Returns
    -------
    Path
        The label file that was written.
    """
    target = annotation_path_for(image_path)
    target.write_text(annotation.to_json(), encoding="utf-8")
    return target


def discover_samples(root: Path) -> list[Sample]:
    """Collect every image sample below *root*, labeled or not.

    Each image is one sample: its labels are read from
    :func:`annotation_path_for` when that file exists, and from an empty
    annotation otherwise. Removed samples (``<round>_<throw>_d``) and hidden
    folders are skipped.

    Parameters
    ----------
    root : Path
        The folder to scan.

    Returns
    -------
    list[Sample]
        The samples in natural path order.
    """
    root = Path(root)
    samples: list[Sample] = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if _is_ignored(path.parent, root) or is_removed_sample(path.parent.name):
            continue
        annotation = _read_annotation(annotation_path_for(path)) or ImageAnnotation()
        samples.append(Sample(annotation, path))

    samples.sort(key=lambda sample: natural_key(sample.image_path))
    return samples


def _read_annotation(path: Path) -> ImageAnnotation | None:
    """Read a label file, returning None when it is missing or not an annotation."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logging.warning(f"Skipping unreadable annotation {path}: {error}")
        return None
    if not isinstance(data, dict) or "throws" not in data:
        return None
    try:
        return ImageAnnotation.from_dict(data)
    except (TypeError, ValueError) as error:
        logging.warning(f"Skipping unusable annotation {path}: {error}")
        return None


def _is_ignored(directory: Path, root: Path) -> bool:
    """Check whether *directory* or one of its parents below *root* is hidden."""
    return any(part.startswith(".") for part in directory.relative_to(root).parts)
