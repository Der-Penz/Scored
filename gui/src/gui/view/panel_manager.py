"""Dock management for the application's panels."""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable, Mapping, Sequence

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import PanelRebound
from gui.model.layout import Column, Layout, Panel, default_layout

DEFAULT_RATIO = 0.5
MIN_RATIO = 0.15
MAX_RATIO = 0.85
COLUMN_PADDING = 5

# A panedwindow only has a sash once it holds at least two panes.
MIN_PANES_FOR_SASH = 2

# ttk's panedwindow has no option for the sash, so its size is measured from the
# gap between the panes. These bound what is believed to be a measurement rather
# than a mid-layout artefact, falling back to ttkbootstrap's usual thickness.
SASH_SIZE = 4
MAX_SASH_SIZE = 16

# The smallest size at which each panel is still usable, as (width, height) in
# pixels. The dartboard draws a fixed square board, the feed needs enough room to
# aim a dart by eye, and the control panel has to fit the scorepad and the turn.
#
# Tk's panedwindow has no minsize option, so these are enforced by clamping the
# sash whenever it moves. See _place_sash.
MIN_PANEL_SIZE: Mapping[Panel, tuple[int, int]] = {
    Panel.DARTBOARD: (320, 320),
    Panel.SOURCE: (240, 160),
    Panel.CONTROL: (280, 260),
}

PanelFactory = Callable[[tk.Misc], ttk.Frame]


def _clamp(value: int, low: int, high: int) -> int:
    """
    Return *value* limited to the inclusive range ``[low, high]``.

    An empty range means there is not enough room to satisfy both ends, so the
    value is returned unchanged rather than collapsing onto one bound.
    """
    if low > high:
        return value
    return max(low, min(high, value))


class _SashGauge:
    """
    Measures and remembers the room a panedwindow's sash takes up.

    ttk's panedwindow has no option for it, so it is worked out from the gap
    between the panes. Only the first measurement is kept, because later ones
    wobble by a pixel as the panes settle and a size that wobbles would make the
    minimum size it feeds back into the window wobble with it.
    """

    def __init__(self, panedwindow: ttk.Panedwindow, extent: str) -> None:
        self._panedwindow = panedwindow
        self._extent = extent
        self._size: int | None = None

    def __call__(self) -> int:
        """Return the size of the sash, or zero while there is no sash."""
        if self._size is not None:
            return self._size
        if len(self._panedwindow.panes()) < MIN_PANES_FOR_SASH:
            return 0

        self._size = self._measure()
        return self._size

    def _measure(self) -> int:
        """Return the gap between the panes, or the ttk default if implausible."""
        measure = f"winfo_{self._extent}"
        panes = [self._panedwindow.nametowidget(name) for name in self._panedwindow.panes()]
        gap = getattr(self._panedwindow, measure)() - sum(
            getattr(pane, measure)() for pane in panes
        )
        return gap if 0 < gap <= MAX_SASH_SIZE else SASH_SIZE


class _Column(ttk.Frame):
    """
    One column of the main window, stacking up to two panels vertically.

    The column owns a vertical panedwindow so that two docked panels can be
    resized against each other; with fewer panels it just fills the space.
    """

    def __init__(self, master: tk.Misc, panels: Mapping[Panel, tuple[int, int]]) -> None:
        super().__init__(master, style="Card.TFrame", padding=COLUMN_PADDING)
        self.slot = ttk.Panedwindow(self, orient=tk.VERTICAL)
        self.slot.pack(fill="both", expand=True)

        self._minimums = panels
        self._docked: tuple[Panel, ...] = ()
        self._ratio = DEFAULT_RATIO
        self._sash = _SashGauge(self.slot, "height")
        self.slot.bind("<ButtonRelease-1>", self._on_sash_release)
        self.slot.bind("<B1-Motion>", self._clamp_sash)
        self.slot.bind("<Configure>", self._on_configure)

    def panes(self) -> list[ttk.Frame]:
        """Return the widgets currently docked in this column, in order."""
        return [self.slot.nametowidget(name) for name in self.slot.panes()]

    @property
    def minimum_size(self) -> tuple[int, int]:
        """
        The smallest size this column is usable at, as (width, height).

        A column stacks its panels, so it needs the widest of their minimum
        widths and the sum of their minimum heights, plus the sash between them
        when there is a split to squeeze them apart. The panels live in the slot
        inside the column's padding, so that padding has to be paid for on top.
        """
        if not self._docked:
            return (0, 0)

        width = max(self._minimums[panel][0] for panel in self._docked)
        height = sum(self._minimums[panel][1] for panel in self._docked)
        if len(self._docked) > 1:
            height += self._sash()

        edge = 2 * COLUMN_PADDING
        return width + edge, height + edge

    def detach(self) -> None:
        """
        Remove every pane from the column without destroying the widgets.

        Tk refuses to add a pane that is already owned by another panedwindow,
        and destroying the panedwindow would take its children with it, so
        panes are released one by one before anything is rearranged.
        """
        for widget in self.panes():
            self.slot.forget(widget)

    def attach(self, panels: Sequence[Panel], widgets: Sequence[ttk.Frame]) -> None:
        """
        Dock *widgets* into the column in the given top to bottom order.

        Parameters
        ----------
        panels : Sequence[Panel]
            Which panel each widget shows, in the same order. The column needs
            this to know how small it may be squeezed.
        widgets : Sequence[ttk.Frame]
            The widgets to dock, one per panel.
        """
        self._docked = tuple(panels)
        for widget in widgets:
            widget.pack_forget()
            self.slot.add(widget)
        self._apply_ratio()

    def apply_ratio(self) -> None:
        """Put the sash back where the user left it."""
        self._apply_ratio()

    def apply_ratio_on_resize(self) -> None:
        """Defer re-placing the sash until the new size has been computed."""
        self.after_idle(self._apply_ratio)

    def _on_configure(self, _event: tk.Event) -> None:
        self.apply_ratio_on_resize()

    def _on_sash_release(self, _event: tk.Event | None = None) -> None:
        """Record where the user dragged the sash so it survives a re-layout."""
        if len(self.panes()) < MIN_PANES_FOR_SASH:
            return

        total = self.slot.winfo_height()
        if total <= 1:
            return

        try:
            position = self.slot.sashpos(0)
        except tk.TclError:
            return

        self._ratio = max(MIN_RATIO, min(MAX_RATIO, position / total))

    def _clamp_sash(self, _event: tk.Event | None = None) -> None:
        """
        Pull the sash back if a drag would push a panel below its minimum.

        This runs on every button motion inside the column, so it leaves the
        sash alone unless it is actually out of bounds.
        """
        if len(self.panes()) < MIN_PANES_FOR_SASH:
            return

        try:
            self._place_sash(self.slot.sashpos(0))
        except tk.TclError:
            pass

    def _apply_ratio(self) -> None:
        if len(self.panes()) < MIN_PANES_FOR_SASH:
            return

        total = self.slot.winfo_height()
        if total <= 1:
            return

        self._place_sash(int(total * self._ratio))

    def _place_sash(self, position: int) -> None:
        """
        Move the sash to *position*, holding each panel at its minimum height.

        Tk's panedwindow will happily collapse a pane to nothing, so the
        position is clamped to leave every docked panel its minimum height. The
        column's padding is already inside the slot, so it does not count here.
        """
        if len(self._docked) < MIN_PANES_FOR_SASH:
            return

        total = self.slot.winfo_height()
        if total <= 1:
            return

        top, bottom = self._docked[0], self._docked[1]
        low = self._minimums[top][1]
        high = total - self._minimums[bottom][1] - self._sash()
        clamped = _clamp(position, low, high)

        try:
            if self.slot.sashpos(0) != clamped:
                self.slot.sashpos(0, clamped)
        except tk.TclError:
            pass


class PanelManager(ttk.Frame):
    """
    Owns the panels of the application and where they are docked.

    Tk cannot reparent a widget, and a panedwindow only accepts panes that are
    its direct children. Moving a panel to a different column therefore means
    recreating its widgets, which is announced with a ``PanelRebound`` event so
    controllers can re-acquire their view and repaint from the model. Reordering
    panels inside a column, and showing or hiding them, reuse the existing
    widgets and are therefore free.
    """

    def __init__(self, master: tk.Misc, event_channel: EventChannel) -> None:
        super().__init__(master, style="Card.TFrame", padding=0)
        self._event_channel = event_channel

        self._panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self._panes.pack(fill="both", expand=True)
        self._columns: dict[Column, _Column] = {
            column: _Column(self._panes, MIN_PANEL_SIZE) for column in Column
        }

        self._factories: dict[Panel, PanelFactory] = {}
        self._widgets: dict[Panel, ttk.Frame] = {}
        self._home: dict[Panel, Column] = {}
        self._rebuilt: set[Panel] = set()
        self._ratio = DEFAULT_RATIO
        self._layout = default_layout()
        self._window_minimum = self._read_window_minimum()
        self._window_chrome: tuple[int, int] | None = None
        self._sash = _SashGauge(self._panes, "width")

        self.bind("<Configure>", self._on_configure)
        self._panes.bind("<ButtonRelease-1>", self._on_sash_release)
        self._panes.bind("<B1-Motion>", self._clamp_sash)
        self._panes.bind("<Configure>", self._on_configure)

    def _read_window_minimum(self) -> tuple[int, int]:
        """
        Return the window minimum as configured before the panels had a say.

        Tk reports an unset minimum as ``(1, 1)``, which is treated as no
        minimum at all so the panels get to decide on their own.
        """
        try:
            width, height = self.winfo_toplevel().wm_minsize()
        except tk.TclError:
            return (0, 0)
        return (width, height) if width > 1 or height > 1 else (0, 0)

    def _read_window_chrome(self) -> tuple[int, int]:
        """
        Return the room the window keeps for its own menu bar and separators.

        The window minimum applies to the whole window, so the panels need their
        own minimum topped up with this much before the window will actually be
        able to hold them. It is measured once, while the window is still at the
        size it was configured with, because the answer does not depend on how
        big the window is.
        """
        window = self.winfo_toplevel()
        return (
            max(0, window.winfo_width() - self.winfo_width()),
            max(0, window.winfo_height() - self.winfo_height()),
        )

    # -- setup ------------------------------------------------------------

    def register(self, panel: Panel, factory: PanelFactory) -> None:
        """
        Teach the manager how to build *panel*.

        Parameters
        ----------
        panel : Panel
            The panel the factory creates.
        factory : Callable[[tk.Misc], ttk.Frame]
            Called with the parent to build the panel into.
        """
        self._factories[panel] = factory

    def build(self, layout: Layout | None = None) -> None:
        """
        Create every registered panel and put them on screen.

        Parameters
        ----------
        layout : Layout | None
            The layout to open with, which is usually the one loaded from disk.
            Defaults to the built-in layout.
        """
        self._layout = layout or default_layout()

        for panel, factory in self._factories.items():
            # A hidden panel still needs a home, because Tk fixes a widget's
            # parent at creation and it cannot be moved to another column later.
            column = self._layout.column_of(panel) or Column.LEFT
            self._widgets[panel] = factory(self._columns[column].slot)
            self._home[panel] = column

        self.apply(self._layout)

    # -- queries ----------------------------------------------------------

    @property
    def layout(self) -> Layout:
        """Return the layout currently on screen."""
        return self._layout

    def widget(self, panel: Panel) -> ttk.Frame:
        """
        Return the live widget of *panel*.

        Raises
        ------
        KeyError
            If the panel has not been built yet, or is not a known panel.
        """
        try:
            return self._widgets[Panel(panel)]
        except (KeyError, ValueError):
            raise KeyError(f"Panel {panel!r} has not been built") from None

    def is_mapped(self, panel: Panel) -> bool:
        """Return whether *panel* is docked and actually on screen."""
        return panel in self._widgets and self._widgets[panel].winfo_ismapped()

    # -- layout -----------------------------------------------------------

    def apply(self, layout: Layout) -> None:
        """
        Put the panels on screen according to *layout*.

        Widgets that stay in the same column are reused as they are; only
        panels that changed column are rebuilt.
        """
        self._layout = layout

        for column in self._columns.values():
            column.detach()

        for column, panels in layout.columns():
            for panel in panels:
                self._ensure_widget(panel, column)

        for column, panels in layout.columns():
            self._columns[column].attach(panels, [self._widgets[panel] for panel in panels])

        self._sync_columns()

        # A pane added while its column was still detached from the outer
        # splitter is laid out against a collapsed size and never recovers, so
        # let the geometry settle before measuring and placing the sashes.
        self.update_idletasks()
        for column in self._columns.values():
            column.apply_ratio()
        self._apply_ratio()
        self._apply_window_minimum()

        if self._rebuilt:
            rebuilt, self._rebuilt = self._rebuilt, set()
            for panel in rebuilt:
                logging.debug(f"Rebuilt panel {panel.value} in a new column")
                self._event_channel.emit(PanelRebound(panel=panel))

    def _ensure_widget(self, panel: Panel, column: Column) -> ttk.Frame:
        """Return the widget for *panel*, rebuilding it if it changed column."""
        widget = self._widgets.get(panel)
        if widget is not None and self._home.get(panel) is column:
            return widget

        if widget is not None:
            widget.destroy()
            del self._widgets[panel]

        widget = self._factories[panel](self._columns[column].slot)
        self._widgets[panel] = widget
        self._home[panel] = column
        self._rebuilt.add(panel)
        return widget

    def _sync_columns(self) -> None:
        """Show exactly the non-empty columns, left column first."""
        active: list[_Column] = [
            self._columns[column] for column, panels in self._layout.columns() if panels
        ]
        current = [self._panes.nametowidget(name) for name in self._panes.panes()]
        if current == active:
            return

        for widget in current:
            self._panes.forget(widget)
        for widget in active:
            self._panes.add(widget)

    @property
    def minimum_size(self) -> tuple[int, int]:
        """
        The smallest size the whole panel area is usable at.

        The columns sit side by side, so their minimum widths add up while the
        height is whichever column needs more of it. Only a real split costs the
        extra pixels of its sash.
        """
        active = [self._columns[column] for column, panels in self._layout.columns() if panels]
        if not active:
            return (0, 0)

        width = sum(column.minimum_size[0] for column in active)
        if len(active) > 1:
            width += self._sash()

        return width, max(column.minimum_size[1] for column in active)

    def _apply_window_minimum(self) -> None:
        """
        Stop the window being shrunk below what the docked panels need.

        Panels are only worth showing at a usable size, so the window refuses to
        go smaller than the current layout can accommodate. The bar falls again
        when a roomier layout is applied, so a demanding layout does not
        permanently raise it.
        """
        minimum_width, minimum_height = self.minimum_size
        base_width, base_height = self._window_minimum
        chrome_width, chrome_height = self._window_chrome or (0, 0)

        self.winfo_toplevel().wm_minsize(
            max(minimum_width + chrome_width, base_width),
            max(minimum_height + chrome_height, base_height),
        )

    # -- sash -------------------------------------------------------------

    def _on_configure(self, _event: tk.Event) -> None:
        """
        Keep the split where the user left it when the window is resized.

        The sashes are placed as a fraction of the available size, so a plain
        resize keeps the same proportion. It is deferred so the new size has
        been computed before it is measured.

        The first configure is also the first chance to measure the room the
        window keeps for its own menu bar, which the minimum size needs and
        which cannot be known before the window has been laid out.
        """
        if self._window_chrome is None:
            self._window_chrome = self._read_window_chrome()
            self._apply_window_minimum()

        self.after_idle(self._apply_ratio)
        for column in self._columns.values():
            column.apply_ratio_on_resize()

    def _on_sash_release(self, _event: tk.Event | None = None) -> None:
        """Record the horizontal sash position so it survives a re-layout."""
        if len(self._panes.panes()) < MIN_PANES_FOR_SASH:
            return

        total = self._panes.winfo_width()
        if total <= 1:
            return

        try:
            position = self._panes.sashpos(0)
        except tk.TclError:
            return

        self._ratio = max(MIN_RATIO, min(MAX_RATIO, position / total))

    def _clamp_sash(self, _event: tk.Event | None = None) -> None:
        """
        Pull the sash back if a drag would push a column below its minimum.

        This runs on every button motion inside the panels, so it leaves the
        sash alone unless it is actually out of bounds.
        """
        if len(self._panes.panes()) < MIN_PANES_FOR_SASH:
            return

        try:
            self._place_sash(self._panes.sashpos(0))
        except tk.TclError:
            pass

    def _apply_ratio(self) -> None:
        if len(self._panes.panes()) < MIN_PANES_FOR_SASH:
            return

        total = self._panes.winfo_width()
        if total <= 1:
            return

        self._place_sash(int(total * self._ratio))

    def _place_sash(self, position: int) -> None:
        """
        Move the sash to *position*, holding each column at its minimum width.

        Both columns share the window horizontally, so the position is clamped to
        leave each of them its minimum width.
        """
        if len(self._panes.panes()) < MIN_PANES_FOR_SASH:
            return

        total = self._panes.winfo_width()
        if total <= 1:
            return

        active = [self._columns[column] for column, panels in self._layout.columns() if panels]
        if len(active) < MIN_PANES_FOR_SASH:
            return

        low = active[0].minimum_size[0]
        high = total - active[-1].minimum_size[0] - self._sash()
        clamped = _clamp(position, low, high)

        try:
            if self._panes.sashpos(0) != clamped:
                self._panes.sashpos(0, clamped)
        except tk.TclError:
            pass
