import glob
import logging
import os
from pathlib import Path

ANNOTATION_FILENAME = "annotation.json"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def find_image(sample_dir: Path) -> Path | None:
    """Find the image stored next to an annotation."""
    for extension in IMAGE_EXTENSIONS:
        image_path = sample_dir / f"image{extension}"
        if image_path.is_file():
            return image_path
    return None


def _absolute_glob(pattern: str) -> tuple[Path, str]:
    """Return the literal root and the absolute pattern to search."""
    path = Path(pattern).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    normalized = Path(os.path.normpath(path))
    for index, part in enumerate(normalized.parts):
        if glob.has_magic(part):
            return Path(*normalized.parts[:index]).resolve(), str(normalized)
    return normalized.resolve(), str(normalized)


def discover_from_path(path: Path) -> tuple[Path, set[Path]]:
    """Collect samples from an existing file or directory."""
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != ANNOTATION_FILENAME:
            raise ValueError(f"{resolved} is not an {ANNOTATION_FILENAME} file")
        return resolved.parent, {resolved.parent}

    samples = {annotation.parent for annotation in resolved.rglob(ANNOTATION_FILENAME)}
    return resolved, samples


def discover_from_glob(pattern: str) -> tuple[Path, set[Path]]:
    """Collect samples from a glob pattern."""
    root, absolute_pattern = _absolute_glob(pattern)
    if not root.is_dir():
        logging.warning("Glob root %s does not exist, skipping %r", root, pattern)
        return root, set()

    samples: set[Path] = set()
    for match in glob.iglob(absolute_pattern, recursive=True):
        path = Path(match)
        if path.is_file():
            if path.name == ANNOTATION_FILENAME:
                samples.add(path.parent)
        else:
            samples.update(annotation.parent for annotation in path.rglob(ANNOTATION_FILENAME))

    return root, samples


def discover(argument: str) -> tuple[Path, set[Path]]:
    """Resolve one command line input into a root and its samples."""
    path = Path(argument).expanduser()
    if path.exists():
        return discover_from_path(path)
    return discover_from_glob(argument)
