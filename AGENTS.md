# AGENTS.md

Single-camera dart scoring: a tkinter GUI, a shared domain library, and a PyTorch
keypoint model.

## Tool usage

- Read and search project files with the built-in read and grep tools, always. Never
  reach for shell utilities such as `grep`, `findstr`, `awk`, `head`, `tail`, `cat`,
  `echo`, or `print` to read or locate project files.
- `tree` is fine to get a file overview.

## Layout

`uv` workspace on Python 3.13 (`.python-version`). Members use `src/` layout:

- `scored_lib/` (`import scored_lib`) — shared domain logic, no GUI/NN deps: game
  rules, dart scoring, geometry, annotation persistence, logging setup. `gui` and
  `nn` depend on it; it must not depend on them.
- `gui/` (`import gui`) — ttkbootstrap/Tkinter app, entrypoint `gui.main:main`.
- `nn/` (`import nn`) — YOLOv10-style keypoint model (timm backbone, PANet neck,
  decoupled head). Model code only, no training entrypoint yet.
- Root `scored` package is a workspace aggregator with no source of its own.

## Commands

Run everything through `uv` from the repo root. Never install packages yourself
(`pip install`, `uv add`, `uv sync`, ...): ask the user to run installs.

- `uv run scored-gui` — start the GUI. Flags: `--source <webcam-index | video-path |
  http-url>`, `--data_dir <path>`, `--log_level DEBUG`.
- `uv run scored-gui --source ./video/test.mp4` — more presets in `.vscode/tasks.json`.
- `uv run ruff check .` / `uv run ruff format .` — lint / format (see Style).
- `uv run pre-commit run --all-files` — run every hook without committing. Hooks are
  installed with `uv run pre-commit install` (ask the user to run installs).

No test suite and no CI.

## Style

- Formatting: `ruff format`, line length 100 (`[tool.ruff]` / `[tool.ruff.format]` in
  the root `pyproject.toml`): double quotes, spaces, docstring code formatting.
  Run `uv run ruff format .`.
- Linting: `ruff check` with `[tool.ruff.lint]` in the root `pyproject.toml`
  (`select = ["E", "W", "F", "I", "B", "UP", "SIM", "LOG", "T10"]`, `ignore = ["LOG015"]`,
  `combine-as-imports = true`). Import sorting, pyupgrade, bugbear and simplify rules
  all come from ruff — there is no separate isort/black. Run `uv run ruff check --fix .`.
- pre-commit (`.pre-commit-config.yaml`) enforces this on every commit: ruff check
  with `--fix`, then ruff format, plus file hygiene (trailing whitespace, final
  newline, LF line endings, yaml/toml checks, no large files, no commits to `main`).
  Hooks run only on staged files.
- Do not add decorative/banner comments (`# ----`, `# =====`, `# Section`) to
  structure the code. Comments explain *why*, they are not headings.
- Do not wrap one-liners in functions, and do not split simple logic into helpers
  that do not earn it. Inline anything that does not need a function.
- Docstrings follow the NumPy style used throughout `scored_lib`.

## GUI architecture

- `main.py` builds `AppModel` -> `AppView` -> `AppController`.
- Controllers extend `BaseController` (`gui/protocols/controller.py`) and communicate
  **only** via the typed `EventChannel` (`gui/events/`); they never hold references to
  each other. Add an event in `gui/events/event_types.py` and subscribe in a controller.
- `AppModel` is the single source of truth for shared state (`players`, `game`, `config`).
- Views are dockable panels (`Panel.DARTBOARD/SOURCE/CONTROL`). Tk cannot reparent a
  widget, so moving/floating a panel rebuilds its widgets; subscribe with
  `BaseController.on_panel_rebound(...)` to re-acquire widgets and repaint from the model.
  Layout persists to `layout.json` in the OS user-state dir.
- `main()` calls `setup_logging()` once; elsewhere just `import logging`.

## Coordinates

- Annotation and training coordinates are normalized **image** coordinates (0..1 over
  the full camera frame), matching the YOLO keypoint labels in `train/data`.
- The scoring math is separate: `scored_lib.dart.scoring` and `RING_RADIUS_NORMALIZED`
  work in the board-relative **scoring frame** — `(0,0)` board center, ~`±1` at the
  edge, angle 0 = center-right increasing clockwise
  (`scored_lib.util.angle.get_clockwise_angle`). Number selection applies a 4-bed
  offset into `DARTBOARD_NUMBERS` (`scored_lib/dart/constants.py`).
- The dartboard canvas uses normalized `0..1` with center `0.5`. Convert between it and
  the scoring frame with `canvas_to_relative_position` / `relative_to_canvas_position`
  (`scored_lib/util/position.py`) before scoring.

## Collected data format

- `LegAnnotationHandler` (`scored_lib/annotation/leg_annotation_handler.py`) owns the
  on-disk layout: `<leg>/leg.json` plus one folder per sample `<round>_<throw>/`
  containing `image.jpg` + `annotation.json` (one-based round/throw).
- Images are held as RGB but written with `cv2` as BGR (`COLOR_RGB2BGR`).
- Removed samples are tombstoned as `<round>_<throw>_d` by default, or deleted,
  depending on the Data menu setting.

## NN / dataset

- `torch`/`torchvision` come from the explicit CPU-only `pytorch-cpu` index in
  `nn/pyproject.toml` — expect no CUDA. (CUDA will later be added for remote training.)
