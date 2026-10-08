from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from dataclasses_json import DataClassJsonMixin, config, dataclass_json
from platformdirs import user_state_dir

APP_NAME = "scored"


class Panel(str, Enum):
    """A dockable view of the application."""

    DARTBOARD = "dartboard"
    SOURCE = "source"
    CONTROL = "control"
    ANNOTATION = "annotation"

    @property
    def label(self) -> str:
        """Human readable name used in menus."""
        return _PANEL_LABELS[self]


class Placement(str, Enum):
    """Where a panel is shown."""

    LEFT = "left"
    RIGHT = "right"
    FLOATING = "floating"
    HIDDEN = "hidden"

    @property
    def label(self) -> str:
        """Human readable name used in menus."""
        return _PLACEMENT_LABELS[self]

    @property
    def is_dock(self) -> bool:
        """Return whether this placement is one of the two window columns."""
        return self in (Placement.LEFT, Placement.RIGHT)


_PANEL_LABELS = {
    Panel.DARTBOARD: "Dartboard",
    Panel.SOURCE: "Camera Feed",
    Panel.CONTROL: "Control Panel",
    Panel.ANNOTATION: "Annotation",
}

_PLACEMENT_LABELS = {
    Placement.LEFT: "Left",
    Placement.RIGHT: "Right",
    Placement.FLOATING: "Floating",
    Placement.HIDDEN: "Hidden",
}

# Where a panel goes when the layout does not say.
DEFAULT_PLACEMENT = Placement.LEFT

# Panels that stay out of the way until the user turns them on, so a saved
# layout from an older version does not suddenly take up room.
_PANEL_DEFAULTS = {
    Panel.ANNOTATION: Placement.HIDDEN,
}


def default_placement_of(panel: Panel) -> Placement:
    """Return where *panel* goes when the layout does not mention it."""
    return _PANEL_DEFAULTS.get(panel, DEFAULT_PLACEMENT)


def _encode_placements(placements: Mapping[Panel, Placement]) -> dict[str, str]:
    """Serialize panel/placement enum mapping into string keys/values for JSON."""
    return {panel.value: placement.value for panel, placement in placements.items()}


def _decode_placements(data: object) -> dict[Panel, Placement]:
    """Deserialize JSON data back into panel/placement mappings, skipping unknowns."""
    if not isinstance(data, dict):
        raise TypeError("layout must be an object")

    placements: dict[Panel, Placement] = {}
    for key, value in data.items():
        try:
            panel = Panel(key)
            placement = Placement(value)
        except ValueError:
            continue
        placements[panel] = placement
    return placements


@dataclass_json
@dataclass(frozen=True)
class Layout(DataClassJsonMixin):
    """
    Where every panel is shown.

    ``placements`` maps every panel to its :class:`Placement`. Entries the
    caller leaves out count as :func:`default_placement_of`, which keeps a
    saved layout usable after a new panel is added.
    """

    placements: Mapping[Panel, Placement] = field(
        default_factory=dict,
        metadata=config(
            encoder=_encode_placements,
            decoder=_decode_placements,
        ),
    )

    def __post_init__(self) -> None:
        given = {
            Panel(panel): Placement(spot)
            for panel, spot in dict(self.placements).items()
        }
        object.__setattr__(
            self,
            "placements",
            {panel: given.get(panel, default_placement_of(panel)) for panel in Panel},
        )

    def placement_of(self, panel: Panel) -> Placement:
        """Return where *panel* is shown."""
        return self.placements[Panel(panel)]

    def is_visible(self, panel: Panel) -> bool:
        """Return whether *panel* is shown at all."""
        return self.placement_of(panel) is not Placement.HIDDEN

    def panels_at(self, placement: Placement) -> tuple[Panel, ...]:
        """Return the panels shown at *placement*, in :class:`Panel` order."""
        return tuple(panel for panel in Panel if self.placement_of(panel) is placement)

    def docks(self) -> Iterator[tuple[Placement, tuple[Panel, ...]]]:
        """Yield the non-empty columns, left one first."""
        for placement in Placement:
            if placement.is_dock:
                panels = self.panels_at(placement)
                if panels:
                    yield placement, panels

    def with_placement(self, panel: Panel, placement: Placement) -> Layout:
        """Return a layout that shows *panel* at *placement*."""
        return replace(
            self,
            placements={**self.placements, Panel(panel): Placement(placement)},
        )


def default_layout() -> Layout:
    """Return the layout used when nothing has been persisted yet."""
    return Layout(
        placements={
            Panel.DARTBOARD: Placement.LEFT,
            Panel.SOURCE: Placement.LEFT,
            Panel.CONTROL: Placement.RIGHT,
        }
    )


def _layout_path() -> Path:
    """Return the file the layout is persisted to."""
    return Path(user_state_dir(appname=APP_NAME, appauthor=False)) / "layout.json"


def load_layout(path: Path | None = None) -> Layout:
    """
    Read the persisted layout, falling back to the default.

    A missing file, unreadable file, malformed JSON or unusable layout all end up
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
        return Layout.from_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logging.warning(f"Ignoring unusable layout in {target}: {exc}")
        return default_layout()


def save_layout(layout: Layout, path: Path | None = None) -> None:
    """Write *layout* to disk, logging but not raising on failure."""
    target = path or _layout_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(layout.to_json(indent=2), encoding="utf-8")
    except OSError as exc:
        logging.warning(f"Could not save layout to {target}: {exc}")
