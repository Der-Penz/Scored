from __future__ import annotations

import logging
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


def session_timestamp(start_time: datetime | None = None) -> str:
    """
    Format *start_time* as the timestamp used in the log file name.

    Parameters
    ----------
    start_time : datetime, optional
        The moment to format, defaults to now.

    Returns
    -------
    str
        A sortable timestamp like ``2026-10-01_18-58-41_512720``.
    """
    return (start_time or datetime.now()).strftime(SESSION_TIME_FORMAT)


def stamp_filename(filename: str, start_time: datetime | None = None) -> str:
    """
    Prefix *filename* with the session timestamp, keeping the extension.

    Parameters
    ----------
    filename : str, optional
        Name of the log file, for example ``"gui_scored.log"``.
    start_time : datetime, optional
        The moment to format, defaults to now.

    Returns
    -------
    str
        For example ``2026-10-01_18-58-41_512720_gui_scored.log``.
    """
    return (
        f"{session_timestamp(start_time)}_{Path(filename).stem}{Path(filename).suffix}"
    )


def list_sessions(base_dir: Path, filename: str) -> list[tuple[str, list[Path]]]:
    """
    Group the log files of *filename* in *base_dir* by their session timestamp.

    Only files carrying a session timestamp are listed, other content of
    *base_dir* is ignored. Rotation backups belong to the session of their log
    file. The sessions are sorted by timestamp, so oldest first.

    Parameters
    ----------
    base_dir : Path
        Directory holding the log files.
    filename : str
        Name of the log file without the timestamp, e.g. ``"scored.log"``.

    Returns
    -------
    list of (str, list[Path])
        The session timestamps with their files.
    """
    marker = f"_{Path(filename).stem}{Path(filename).suffix}"
    sessions: dict[str, list[Path]] = {}

    for path in base_dir.glob(f"*{marker}*"):
        timestamp, separator, backup = path.name.partition(marker)
        if not separator or (backup and not backup.replace(".", "").isdigit()):
            continue
        try:
            datetime.strptime(timestamp, SESSION_TIME_FORMAT)
        except ValueError:
            continue
        sessions.setdefault(timestamp, []).append(path)

    return sorted(sessions.items())


def prune_sessions(
    base_dir: Path, filename: str, keep: int = DEFAULT_MAX_SESSIONS
) -> list[Path]:
    """
    Delete the oldest session logs so at most *keep* of them remain.

    The running log is never removed.

    Parameters
    ----------
    base_dir : Path
        Directory holding the log files.
    filename : str
        Name of the log file without the timestamp.
    keep : int, optional
        Number of sessions to keep, by default 20. Values below one keep the
        newest session.

    Returns
    -------
    list[Path]
        The removed files, oldest session first.
    """
    keep = max(1, keep)
    sessions = list_sessions(base_dir, filename)

    removed: list[Path] = []
    for _, paths in sessions[: max(0, len(sessions) - keep)]:
        for path in paths:
            path.unlink(missing_ok=True)
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
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    start_time: datetime | None = None,
) -> Path | None:
    """
    Configure the root logger with a rotating file and an optional console handler.

    The log file name is prefixed with the start time, so every start writes its
    own log file and nothing is overwritten. Calling this function again
    replaces the handlers installed by a previous call instead of stacking them.

    Parameters
    ----------
    log_dir : Path, optional
        Base directory for the log file, defaults to ``./logs``.
    filename : str, optional
        Name of the log file without the timestamp, defaults to ``"scored.log"``.
    level : int | str, optional
        Initial log level, defaults to ``"INFO"``.
    console : bool, optional
        Also log to stderr, by default True.
    stream : TextIO, optional
        Alternative stream for the console handler, by default stderr.
    max_bytes, backup_count : int, optional
        Rotation limits of the file handler.
    max_sessions : int, optional
        Number of log files to keep, by default 20. The oldest ones are deleted,
        values below one keep the newest one.
    start_time : datetime, optional
        Timestamp in the log file name, defaults to now.

    Returns
    -------
    Path or None
        The path of the log file, or None when the file handler is unavailable.
    """
    global _LOG_FILE

    # Release the handlers of a previous call first, otherwise their log files
    # stay open and cannot be removed when old logs are pruned.
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
    removed_logs: list[Path] = []
    base_dir = Path(log_dir) if log_dir is not None else Path.cwd() / DEFAULT_LOG_DIR

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        log_file = base_dir / stamp_filename(filename, start_time)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

        # Pruned after the new file exists, so it counts towards the limit and
        # the number of log files never exceeds max_sessions.
        removed_logs = prune_sessions(base_dir, filename, max_sessions)
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
        logging.warning(
            f"File logging disabled, cannot write to {base_dir}: {file_error}"
        )

    for log in removed_logs:
        logging.info(f"Removed old log file {log.name}")

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
