"""Persisted description of which views are visible and where they are docked."""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from platformdirs import user_state_dir

APP_NAME = "scored"
LAYOUT_FILENAME = "layout.json"
LAYOUT_VERSION = 1

MAX_PANELS_PER_COLUMN = 2


class Panel(str, Enum):
    """A dockable view of the application."""

    DARTBOARD = "dartboard"
    SOURCE = "source"
    CONTROL = "control"

    @property
    def label(self) -> str:
        """Human readable name used in menus."""
        return _PANEL_LABELS[self]


class Column(str, Enum):
    """One of the two dock columns of the main window."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def label(self) -> str:
        """Human readable name used in menus."""
        return "Left Column" if self is Column.LEFT else "Right Column"


_PANEL_LABELS = {
    Panel.DARTBOARD: "Dartboard",
    Panel.SOURCE: "Camera Feed",
    Panel.CONTROL: "Control Panel",
}


class LayoutError(ValueError):
    """Raised when a layout would put a panel in an impossible place."""


def _as_panels(panels: Iterable[object]) -> tuple[Panel, ...]:
    """Coerce *panels* into a tuple of Panel, rejecting anything unknown."""
    resolved: list[Panel] = []
    for panel in panels:
        try:
            member = Panel(panel)
        except ValueError:
            raise LayoutError(f"Unknown panel: {panel!r}") from None
        if member not in resolved:
            resolved.append(member)
    return tuple(resolved)


@dataclass(frozen=True)
class Layout:
    """
    Which panels are docked in which column, and which are hidden.

    ``left``, ``right`` and ``hidden`` partition :class:`Panel`: every panel is
    in exactly one of them, and each column holds at most
    :data:`MAX_PANELS_PER_COLUMN` panels. ``last_column`` is a non-authoritative
    hint remembering where a hidden panel used to be docked, so that showing it
    again can put it back where the user last had it.
    """

    left: tuple[Panel, ...] = (Panel.DARTBOARD, Panel.SOURCE)
    right: tuple[Panel, ...] = (Panel.CONTROL,)
    hidden: frozenset[Panel] = frozenset()
    last_column: Mapping[Panel, Column] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise and validate the layout."""
        object.__setattr__(self, "left", _as_panels(self.left))
        object.__setattr__(self, "right", _as_panels(self.right))
        object.__setattr__(self, "hidden", frozenset(Panel(p) for p in self.hidden))
        object.__setattr__(
            self,
            "last_column",
            {Panel(p): Column(c) for p, c in dict(self.last_column).items()},
        )

        for column in Column:
            panels = self.column(column)
            if len(panels) > MAX_PANELS_PER_COLUMN:
                raise LayoutError(
                    f"{column.label} can hold at most {MAX_PANELS_PER_COLUMN} panels, "
                    f"got {len(panels)}"
                )

        placed = self.left + self.right
        duplicates = {p for p in placed if placed.count(p) > 1}
        if duplicates:
            raise LayoutError(
                "Panels cannot be docked twice: " + ", ".join(sorted(p.label for p in duplicates))
            )

        overlap = set(placed) & set(self.hidden)
        if overlap:
            raise LayoutError(
                "Panels cannot be docked and hidden at once: "
                + ", ".join(sorted(p.label for p in overlap))
            )

        missing = set(Panel) - set(placed) - set(self.hidden)
        if missing:
            raise LayoutError(
                "Panels must be docked or hidden: " + ", ".join(sorted(p.label for p in missing))
            )

    # -- queries ----------------------------------------------------------

    def column(self, column: Column) -> tuple[Panel, ...]:
        """Return the panels docked in *column*, in top to bottom order."""
        return self.left if column is Column.LEFT else self.right

    def columns(self) -> Iterator[tuple[Column, tuple[Panel, ...]]]:
        """Yield every column with its docked panels, left column first."""
        for column in Column:
            yield column, self.column(column)

    def is_visible(self, panel: Panel) -> bool:
        """Return whether *panel* is currently shown."""
        return panel not in self.hidden

    def column_of(self, panel: Panel) -> Column | None:
        """Return the column *panel* is docked in, or None while hidden."""
        for column in Column:
            if panel in self.column(column):
                return column
        return None

    def panels_in(self, column: Column) -> Iterator[Panel]:
        """Yield the panels docked in *column*."""
        return iter(self.column(column))

    # -- transitions ------------------------------------------------------

    def with_panel(self, panel: Panel, column: Column | None) -> Layout:
        """
        Return a layout with *panel* docked in *column*, or hidden if it is None.

        The panel is removed from wherever it currently is. Docking a panel
        into a column that is already full puts the panel that occupied that
        column last into the hidden set instead of raising, so the menu can
        never end up in a state it cannot represent.
        """
        panel = Panel(panel)
        left = [p for p in self.left if p is not panel]
        right = [p for p in self.right if p is not panel]
        hidden = set(self.hidden)
        hidden.discard(panel)
        last_column = dict(self.last_column)

        if column is None:
            if self.column_of(panel) is not None:
                last_column[panel] = Column(self.column_of(panel))
            hidden.add(panel)
            return replace(
                self,
                left=tuple(left),
                right=tuple(right),
                hidden=frozenset(hidden),
                last_column=last_column,
            )

        target = left if column is Column.LEFT else right
        if len(target) >= MAX_PANELS_PER_COLUMN:
            evicted = target.pop()
            hidden.add(evicted)
            last_column[evicted] = column

        target.append(panel)
        last_column[panel] = column
        return replace(
            self,
            left=tuple(left),
            right=tuple(right),
            hidden=frozenset(hidden),
            last_column=last_column,
        )

    def with_column(self, column: Column, panels: Sequence[Panel]) -> Layout:
        """
        Return a layout whose *column* holds exactly *panels*, top to bottom.

        Panels named here are moved into *column*, taking them out of the other
        column so they are never docked twice. Anything else that was docked and
        is not named here is hidden rather than relocated, so choosing a column
        never invents a placement for a panel the user did not ask for.
        """
        ordered = _as_panels(panels)
        if len(ordered) > MAX_PANELS_PER_COLUMN:
            raise LayoutError(f"{column.label} can hold at most {MAX_PANELS_PER_COLUMN} panels")

        other = Column.RIGHT if column is Column.LEFT else Column.LEFT
        kept = tuple(p for p in self.column(other) if p not in ordered)
        docked = set(ordered) | set(kept)
        hidden = frozenset(p for p in Panel if p not in docked)
        last_column = dict(self.last_column)
        for panel in ordered:
            last_column[panel] = column

        if column is Column.LEFT:
            return replace(
                self,
                left=ordered,
                right=kept,
                hidden=hidden,
                last_column=last_column,
            )
        return replace(
            self,
            left=kept,
            right=ordered,
            hidden=hidden,
            last_column=last_column,
        )

    def toggle(self, panel: Panel) -> Layout:
        """Return a layout that flips the visibility of *panel*."""
        panel = Panel(panel)
        if self.is_visible(panel):
            return self.with_panel(panel, None)
        return self.with_panel(panel, self.last_column.get(panel) or Column.LEFT)

    def swap_columns(self) -> Layout:
        """Return a layout with the left and right columns exchanged."""
        return replace(self, left=self.right, right=self.left)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON serialisable representation of the layout."""
        return {
            "version": LAYOUT_VERSION,
            "left": [p.value for p in self.left],
            "right": [p.value for p in self.right],
            "hidden": sorted(p.value for p in self.hidden),
            "last_column": {p.value: c.value for p, c in self.last_column.items()},
        }

    @classmethod
    def from_dict(cls, data: object) -> Layout:
        """Build a layout from :meth:`to_dict` output, rejecting bad shapes."""
        if not isinstance(data, dict):
            raise LayoutError("layout must be an object")

        last_column = data.get("last_column", {})
        if not isinstance(last_column, dict):
            raise LayoutError("last_column must be an object")

        return cls(
            left=tuple(data.get("left", ()) or ()),
            right=tuple(data.get("right", ()) or ()),
            hidden=frozenset(data.get("hidden", ()) or ()),
            last_column=last_column,
        )


def default_layout() -> Layout:
    """Return the layout used when nothing has been persisted yet."""
    return Layout(
        left=(Panel.DARTBOARD, Panel.SOURCE),
        right=(Panel.CONTROL,),
        hidden=frozenset(),
        last_column={
            Panel.DARTBOARD: Column.LEFT,
            Panel.SOURCE: Column.LEFT,
            Panel.CONTROL: Column.RIGHT,
        },
    )


def column_options() -> list[tuple[str, str, tuple[Panel, ...] | None]]:
    """
    Return the (key, label, panels) choices offered for a column.

    A column may be empty, hold a single panel, or stack any pair of panels.
    The key is stable and is what a menu stores in its radio variable; an empty
    key means the column is not shown.
    """
    options: list[tuple[str, str, tuple[Panel, ...] | None]] = [("", "None", None)]
    for size in range(1, MAX_PANELS_PER_COLUMN + 1):
        for combo in itertools.combinations(Panel, size):
            key = "+".join(panel.value for panel in combo)
            label = " + ".join(panel.label for panel in combo)
            options.append((key, label, combo))
    return options


def column_key(panels: Sequence[Panel]) -> str:
    """Return the option key describing exactly *panels*."""
    return "+".join(panel.value for panel in panels)


def _layout_path() -> Path:
    """Return the file the layout is persisted to."""
    return Path(user_state_dir(appname=APP_NAME, appauthor=False)) / LAYOUT_FILENAME


def load_layout(path: Path | None = None) -> Layout:
    """
    Read the persisted layout, falling back to the default.

    A missing file, unreadable file, malformed JSON or invalid layout all end up
    as the default layout, because a broken layout must never stop the app from
    starting.
    """
    target = path or _layout_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.debug(f"No layout file at {target}, using the default layout")
        return default_layout()
    except OSError as exc:
        logging.warning(f"Could not read layout from {target}: {exc}")
        return default_layout()

    try:
        return Layout.from_dict(json.loads(raw))
    except (json.JSONDecodeError, LayoutError, TypeError, ValueError) as exc:
        logging.warning(f"Ignoring unusable layout in {target}: {exc}")
        return default_layout()


def save_layout(layout: Layout, path: Path | None = None) -> None:
    """Write *layout* to disk, logging but not raising on failure."""
    target = path or _layout_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(layout.to_dict(), indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.warning(f"Could not save layout to {target}: {exc}")
