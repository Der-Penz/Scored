"""Logging setup shared by the scored packages.

:func:`setup_logging` configures the root logger, so every module can simply
``import logging`` and call the module level functions like
``logging.info("...")`` without creating named loggers first::

    import logging
    from scored_lib.logging_setup import setup_logging

    setup_logging()          # writes ./logs/<start time>/scored.log and echoes to stderr
    logging.info("started")  # goes to both

Every start gets its own timestamped folder so a restart never overwrites an
earlier log. The oldest of those folders are removed once more than
``max_sessions`` of them exist.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILENAME = "scored.log"
DEFAULT_LEVEL = "INFO"
DEFAULT_FORMAT = "%(asctime)s [%(levelname)-8s] %(filename)s:%(lineno)-4d %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_BACKUP_COUNT = 5
DEFAULT_MAX_SESSIONS = 20
SESSION_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S_%f"

# Third party loggers that are too chatty for the file handler.
QUIET_LOGGERS = ("cv2", "urllib3", "PIL")

_LOG_FILE: Path | None = None


def _resolve_level(level: int | str) -> int:
    """
    Convert *level* into a logging level number.

    Parameters
    ----------
    level : int | str
        A level number like ``logging.INFO`` or a level name like ``"info"``.
        Unknown names fall back to ``DEFAULT_LEVEL``.

    Returns
    -------
    int
        The numeric logging level.
    """
    if isinstance(level, int):
        return level

    resolved = logging.getLevelNamesMapping().get(str(level).upper())
    if not isinstance(resolved, int):
        logging.warning(f"Unknown log level {level!r}, falling back to {DEFAULT_LEVEL}")
        resolved = logging.getLevelNamesMapping()[DEFAULT_LEVEL]
    return resolved


def resolve_caller(frame: object | None = None) -> tuple[str, int]:
    """
    Find the first stack frame outside of the logging package.

    Parameters
    ----------
    frame : object, optional
        The frame to start walking up from, defaults to the caller of this
        function.

    Returns
    -------
    tuple[str, int]
        The file name and line number of the actual caller, or
        ``("<unknown>", 0)`` when the stack runs out.
    """
    logging_package = str(Path(logging.__file__).parent)
    frame = sys._getframe(1) if frame is None else frame

    while frame is not None:
        filename = frame.f_code.co_filename  # type: ignore[attr-defined]
        if not str(Path(filename).parent).startswith(logging_package):
            return Path(filename).name, frame.f_lineno  # type: ignore[attr-defined]
        frame = frame.f_back  # type: ignore[attr-defined]

    return "<unknown>", 0


class CallerFormatter(logging.Formatter):
    """
    Formatter that reports the caller for module level logging calls.

    ``logging.info(...)`` is invoked from inside the logging package, so the
    built in ``%(filename)s`` / ``%(lineno)d`` fields would always point at
    ``logging/__init__.py``. This formatter walks the stack itself and uses the
    first frame outside of the logging package instead.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format *record*, replacing its location with the real caller."""
        record.filename, record.lineno = resolve_caller(sys._getframe(1))
        return super().format(record)


def create_formatter(
    fmt: str | None = None, datefmt: str | None = None, *, resolve_location: bool = True
) -> logging.Formatter:
    """
    Create a formatter matching the format used by :func:`setup_logging`.

    Parameters
    ----------
    fmt, datefmt : str, optional
        Overrides for the default format strings.
    resolve_location : bool, optional
        Look up the caller of every record, by default True. Handlers that
        intercept records should resolve the location themselves with
        :func:`resolve_caller` and set this to False.

    Returns
    -------
    logging.Formatter
        The configured formatter.
    """
    if not resolve_location:
        return logging.Formatter(
            fmt if fmt is not None else DEFAULT_FORMAT,
            datefmt if datefmt is not None else DEFAULT_DATE_FORMAT,
        )

    return CallerFormatter(
        fmt if fmt is not None else DEFAULT_FORMAT,
        datefmt if datefmt is not None else DEFAULT_DATE_FORMAT,
    )


def is_session_folder(path: Path) -> bool:
    """
    Check whether *path* is a folder created by a previous logging session.

    Parameters
    ----------
    path : Path
        The folder to check.

    Returns
    -------
    bool
        True if the folder name is a session timestamp.
    """
    try:
        datetime.strptime(path.name, SESSION_TIME_FORMAT)
    except ValueError:
        return False
    return True


def prune_sessions(base_dir: Path, keep: int = DEFAULT_MAX_SESSIONS) -> list[Path]:
    """
    Delete the oldest session folders so at most *keep* of them remain.

    Only folders named after a session timestamp are considered, other content
    of *base_dir* is left alone. The current session is never removed.

    Parameters
    ----------
    base_dir : Path
        Directory holding the session folders.
    keep : int, optional
        Number of session folders to keep, by default 20. Values below one keep
        the newest folder.

    Returns
    -------
    list[Path]
        The removed folders, oldest first.
    """
    keep = max(1, keep)
    sessions = sorted(path for path in base_dir.glob("*") if is_session_folder(path))

    removed: list[Path] = []
    for path in sessions[: max(0, len(sessions) - keep)]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)

    return removed


def setup_logging(
    log_dir: Path | None = None,
    filename: str = DEFAULT_LOG_FILENAME,
    level: int | str = DEFAULT_LEVEL,
    *,
    console: bool = True,
    stream: TextIO | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    session_folder: bool = True,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    start_time: datetime | None = None,
) -> Path | None:
    """
    Configure the root logger with a rotating file and an optional console handler.

    Calling this function again replaces the handlers installed by a previous
    call instead of stacking them.

    Parameters
    ----------
    log_dir : Path, optional
        Base directory for the log file, defaults to ``./logs``.
    filename : str, optional
        Name of the log file, defaults to ``"scored.log"``.
    level : int | str, optional
        Initial log level, defaults to ``"INFO"``.
    console : bool, optional
        Also log to stderr, by default True.
    stream : TextIO, optional
        Alternative stream for the console handler, by default stderr.
    max_bytes, backup_count : int, optional
        Rotation limits of the file handler.
    session_folder : bool, optional
        Write into a timestamped subfolder so every start keeps its own log,
        by default True.
    max_sessions : int, optional
        Number of session folders to keep, by default 20. The oldest ones are
        deleted, values below one keep the newest folder.
    start_time : datetime, optional
        Timestamp of the session folder, defaults to now.

    Returns
    -------
    Path or None
        The path of the log file, or None when the file handler is unavailable.
    """
    global _LOG_FILE

    # Release the handlers of a previous call first, otherwise their log files
    # stay open and cannot be removed when old sessions are pruned.
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_scored_managed", False):
            root.removeHandler(handler)
            handler.close()

    formatter = create_formatter()
    handlers: list[logging.Handler] = []

    if console:
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    log_file: Path | None = None
    file_error: OSError | None = None
    removed_sessions: list[Path] = []
    base_dir = Path(log_dir) if log_dir is not None else Path.cwd() / DEFAULT_LOG_DIR

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        directory = (
            base_dir / (start_time or datetime.now()).strftime(SESSION_TIME_FORMAT)
            if session_folder
            else base_dir
        )
        directory.mkdir(parents=True, exist_ok=True)

        if session_folder:
            removed_sessions = prune_sessions(base_dir, max_sessions)

        log_file = directory / filename
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except OSError as error:
        log_file = None
        file_error = error

    resolved_level = _resolve_level(level)
    root.setLevel(resolved_level)
    for handler in handlers:
        handler.setLevel(resolved_level)
        handler._scored_managed = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    if file_error is not None:
        logging.warning(f"File logging disabled, cannot write to {base_dir}: {file_error}")

    for session in removed_sessions:
        logging.info(f"Removed old log session {session.name}")

    _LOG_FILE = log_file
    return log_file


def set_log_level(level: int | str) -> int:
    """
    Change the level of the root logger and its managed handlers at runtime.

    Parameters
    ----------
    level : int | str
        The new level, as a number or a name.

    Returns
    -------
    int
        The applied numeric level.
    """
    resolved = _resolve_level(level)
    root = logging.getLogger()
    root.setLevel(resolved)

    for handler in root.handlers:
        if getattr(handler, "_scored_managed", False):
            handler.setLevel(resolved)

    return resolved


def get_log_file() -> Path | None:
    """
    Get the log file used by the last :func:`setup_logging` call.

    Returns
    -------
    Path or None
        The log file path, or None when file logging is unavailable.
    """
    return _LOG_FILE
