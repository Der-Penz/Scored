from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import tyro
from dataset.dataset_util import ANNOTATION_FILENAME, IMAGE_EXTENSIONS, discover, find_image
from dataset.splitting import SPLITTERS
from nn.data.dataset import DatasetConfig
from scored_lib.logging_setup import setup_logging

# Fixed for the current dart keypoint model, so they are not exposed on the CLI.
NUM_KEYPOINTS = 2
CLS_DICT = {"dart": 1}

_REMOVED_SAMPLE_DIR = re.compile(r"\d+_\d+_d+")


@dataclass
class CliArgs:
    """Create a DatasetConfig JSON from annotated leg directories, files or glob patterns."""

    paths: Annotated[
        list[str],
        tyro.conf.Positional,
        tyro.conf.arg(
            help=(
                "Directory, annotation.json file or glob pattern. Patterns support '*', '?' and "
                "'**'; quote them so the shell does not expand them first."
            ),
        ),
    ]
    """One or more directories, annotation files, or glob patterns."""

    description: str
    """Free text description stored in the dataset config."""

    output_dir: Path
    """Folder the dataset config JSON is written to."""

    name: str = "dart_keypoints"
    """Dataset name, also used as the output file name."""

    val_ratio: float = 0.15
    """Fraction of samples used for validation."""

    test_ratio: float = 0.15
    """Fraction of samples used for testing."""

    seed: int = 42
    """Random seed of the train/val/test split."""

    splitter: str = "random"
    """Splitting strategy (e.g. random, stratified)."""

    log_level: str = "INFO"
    """Logging level (e.g. INFO, DEBUG, WARNING)."""

    def __post_init__(self) -> None:
        if not 0 <= self.val_ratio < 1:
            raise ValueError("--val-ratio must be in [0, 1)")
        if not 0 <= self.test_ratio < 1:
            raise ValueError("--test-ratio must be in [0, 1)")
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("--val-ratio and --test-ratio must add up to less than 1")
        if Path(self.name).name != self.name:
            raise ValueError("--name must not contain path separators")


def main(args: CliArgs | None = None) -> None:
    """Build the dataset config and write it to the output folder."""
    if args is None:
        args = tyro.cli(CliArgs)

    setup_logging(filename="dataset_generation.log", level=args.log_level)

    input_roots: list[Path] = []
    samples: set[Path] = set()

    for argument in args.paths:
        try:
            root, discovered = discover(argument)
        except (OSError, ValueError) as err:
            raise SystemExit(str(err)) from err

        input_roots.append(root)
        samples |= discovered
        logging.info(f"{argument}: {len(discovered)} sample(s)")
        if not discovered and root.is_dir():
            logging.warning(f"No {ANNOTATION_FILENAME} files found under {root}")

    if not samples:
        raise SystemExit(f"No {ANNOTATION_FILENAME} files found for the given paths and patterns")

    valid_samples: list[Path] = []
    for sample in sorted(samples):
        if find_image(sample) is not None:
            valid_samples.append(sample)
        elif not _REMOVED_SAMPLE_DIR.fullmatch(sample.name):
            logging.warning(f"Skipping {sample}: no image{'/'.join(IMAGE_EXTENSIONS)} found")

    if not valid_samples:
        raise SystemExit("Every discovered sample is missing its image")

    try:
        base = Path(os.path.commonpath([str(path) for path in [*input_roots, *valid_samples]]))
    except ValueError as err:
        raise SystemExit(f"The given paths do not share a common base path: {err}") from err

    try:
        splitter = SPLITTERS[args.splitter](args.seed, args.val_ratio, args.test_ratio)
        train, val, test = splitter.split_samples(valid_samples)
        train_paths = sorted(sample.relative_to(base).as_posix() for sample in train)
        val_paths = sorted(sample.relative_to(base).as_posix() for sample in val)
        test_paths = sorted(sample.relative_to(base).as_posix() for sample in test)
    except ValueError as err:
        raise SystemExit(str(err)) from err

    config = DatasetConfig(
        name=args.name,
        description=args.description,
        num_keypoints=NUM_KEYPOINTS,
        cls_dict=CLS_DICT,
        path=base,
        train=train_paths,
        val=val_paths,
        test=test_paths,
    )

    payload = asdict(config)
    payload["path"] = config.path.as_posix()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.name}.json"
    output_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    logging.info(
        "Wrote %s with %d train, %d val and %d test sample(s), base path %s",
        output_file,
        len(config.train),
        len(config.val),
        len(config.test),
        config.path,
    )


if __name__ == "__main__":
    main()
