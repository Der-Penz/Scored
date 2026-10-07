"""Dock management for the application's panels."""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable, Mapping

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import PanelClosed, PanelRebound
from gui.model.layout import (
    DEFAULT_PLACEMENT,
    Layout,
    Panel,
    Placement,
    default_layout,
)
from gui.view.top_level_view import TopLevelView

PADDING = 5

# ttk's panedwindow has no option for the sash, so its thickness is only known
# to its layout engine. Four pixels is its usual value and close enough for the
# minimum window size.
SASH = 4

# The smallest size at which a panel is still usable, as (width, height) in
# pixels. The dartboard draws a fixed square board, the feed needs enough room
# to aim a dart by eye, and the control panel has to fit the scorepad and the
# turn.
MIN_PANEL_SIZE: Mapping[Panel, tuple[int, int]] = {
    Panel.DARTBOARD: (320, 320),
    Panel.SOURCE: (240, 160),
    Panel.CONTROL: (280, 260),
}

# Floor for the window, so the menu bar and separators never squeeze the panels.
MIN_WINDOW_SIZE = (400, 300)

FLOAT_SIZE = "640x480"

# How a panel gets built: given the parent to build into, return the widget.
# The view classes themselves are the factories, so they only take a master.
PanelFactory = Callable[[tk.Misc], tk.Misc]


class _Dock(ttk.Frame):
    """
    One column of the main window, stacking its panels vertically.

    Panels share the height with ``pack`` rather than a nested panedwindow, so
    showing or hiding one only has to pack it or forget it. Tk fixes a widget's
    parent at creation, so moving one to the other column means rebuilding it
    into that column.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, style="Card.TFrame", padding=PADDING)
        self._docked: tuple[Panel, ...] = ()

    def dock(self, panels: tuple[Panel, ...], widgets: list[tk.Misc]) -> None:
        """Show *widgets* in this column, in the given top to bottom order."""
        self._docked = panels
        for widget in widgets:
            widget.pack_forget()  # type: ignore
            widget.pack(fill="both", expand=True)  # type: ignore

    def clear(self) -> None:
        """Take every panel out of this column without destroying it."""
        for widget in self.children.values():
            widget.pack_forget()
        self._docked = ()

    @property
    def minimum_size(self) -> tuple[int, int]:
        """
        The smallest size this column is usable at, as (width, height).

        The panels share the height, so the column needs the widest of their
        minimum widths and the tallest of their minimum heights, plus its own
        padding on both sides. Only a column with panels in it is ever measured.
        """
        if not self._docked:
            return (0, 0)

        edge = 2 * PADDING
        return (
            max(MIN_PANEL_SIZE[panel][0] for panel in self._docked) + edge,
            max(MIN_PANEL_SIZE[panel][1] for panel in self._docked) + edge,
        )


class _FloatWindow(TopLevelView):
    """The window a floating panel lives in."""

    GEOMETRY = FLOAT_SIZE
    DIALOG = False

    def __init__(self, master: tk.Misc, panel: Panel) -> None:
        super().__init__(master)
        self.title(panel.label)


class PanelManager(ttk.Frame):
    """
    Owns the panels of the application and where they are shown.

    The panel area is one horizontal panedwindow holding the non-empty columns,
    so the split between them can be dragged but never squeezed below what the
    panels need, and a single column fills the whole area on its own.

    Hiding a panel and showing it again reuses its widgets, because that only
    means packing or forgetting them. Changing which column a panel is in, or
    floating it, means giving it a different parent, which Tk cannot do to an
    existing widget; the panel is built again and announced with a
    ``PanelRebound`` event so controllers can re-acquire their view.
    """

    def __init__(
        self,
        master: tk.Misc,
        event_channel: EventChannel,
        factories: Mapping[Panel, PanelFactory],
    ) -> None:
        super().__init__(master, style="Card.TFrame", padding=0)
        self._event_channel = event_channel

        self._panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self._panes.pack(fill="both", expand=True)
        self._docks = {spot: _Dock(self._panes) for spot in Placement if spot.is_dock}

        self._factories = dict(factories)
        self._widgets: dict[Panel, tk.Misc] = {}
        self._floaters: dict[Panel, tk.Toplevel] = {}
        self._rebuilt: list[Panel] = []
        self._layout = default_layout()

        self._window_minimum = self._read_window_minimum()
        self._window_chrome: tuple[int, int] | None = None

        self.bind("<Configure>", self._on_configure)
        self._panes.bind("<B1-Motion>", self._clamp_sash)

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
        able to hold them.
        """
        window = self.winfo_toplevel()
        return (
            max(0, window.winfo_width() - self.winfo_width()),
            max(0, window.winfo_height() - self.winfo_height()),
        )

    # -- queries ----------------------------------------------------------

    @property
    def layout(self) -> Layout:
        """Return the layout currently on screen."""
        return self._layout

    def widget(self, panel: Panel) -> tk.Misc:
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

    # -- layout -----------------------------------------------------------

    def apply(self, layout: Layout) -> None:
        """Put the panels on screen according to *layout*."""
        self._layout = layout
        self._sync_floaters()

        for dock in self._docks.values():
            dock.clear()

        for panel in self._factories:
            self._widget_for(panel)

        for placement, panels in layout.docks():
            self._docks[placement].dock(
                panels, [self._widgets[panel] for panel in panels]
            )

        for panel in layout.panels_at(Placement.FLOATING):
            self._widgets[panel].pack(
                fill="both", expand=True, padx=PADDING, pady=PADDING
            )  # type: ignore

        self._sync_columns()

        # A pane added while the panedwindow is still measuring itself is laid
        # out against a collapsed size and never recovers, so let the geometry
        # settle before the minimum size is worked out.
        self.update_idletasks()
        self._apply_window_minimum()

        if self._rebuilt:
            rebuilt, self._rebuilt = self._rebuilt, []
            for panel in rebuilt:
                logging.debug(f"Rebuilt panel {panel.value} for a new host")
                self._event_channel.emit(PanelRebound(panel=panel))

    def _host_of(self, panel: Panel) -> Placement:
        """
        Return the host the panel's widget has to be built into.

        A hidden panel is built into the default column without being packed, so
        it still has a parent and its controllers still have widgets to point
        at. Bringing it back to that column is then free.
        """
        placement = self._layout.placement_of(panel)
        return DEFAULT_PLACEMENT if placement is Placement.HIDDEN else placement

    def _widget_for(self, panel: Panel) -> tk.Misc:
        """Return the widget of *panel*, rebuilding it if its host changed."""
        parent = self._parent_for(panel, self._host_of(panel))
        widget = self._widgets.get(panel)
        if widget is not None:
            if widget.master is parent:
                return widget
            widget.destroy()
            del self._widgets[panel]
            self._rebuilt.append(panel)

        widget = self._factories[panel](parent)
        self._widgets[panel] = widget
        return widget

    def _parent_for(self, panel: Panel, placement: Placement) -> tk.Misc:
        """Return the widget that has to own *panel* while it is at *placement*."""
        if placement is Placement.FLOATING:
            return self._floaters[panel]
        return self._docks[placement]

    def _sync_columns(self) -> None:
        """Show exactly the non-empty columns, left one first."""
        current = [self._panes.nametowidget(name) for name in self._panes.panes()]
        wanted = [self._docks[placement] for placement, _ in self._layout.docks()]
        if current == wanted:
            return

        for widget in current:
            self._panes.forget(widget)
        for widget in wanted:
            self._panes.add(widget)

    def _sync_floaters(self) -> None:
        """Open a window per floating panel and close the ones no longer needed."""
        for panel in self._factories:
            floating = self._layout.placement_of(panel) is Placement.FLOATING
            if floating and panel not in self._floaters:
                self._open_floater(panel)
            elif not floating and panel in self._floaters:
                self._close_floater(panel)

    def _open_floater(self, panel: Panel) -> None:
        """Open the window a floating panel lives in."""
        window = _FloatWindow(self.winfo_toplevel(), panel)
        window.protocol(
            "WM_DELETE_WINDOW",
            lambda p=panel: self._event_channel.emit(PanelClosed(panel=p)),
        )
        self._floaters[panel] = window

    def _close_floater(self, panel: Panel) -> None:
        """Close a floating panel's window, which takes the panel down with it."""
        self._floaters.pop(panel).destroy()
        if panel in self._widgets:
            # Closing the window destroyed the panel with it, so it has to be
            # built again for whichever host comes next.
            del self._widgets[panel]
            self._rebuilt.append(panel)

    # -- sizing ------------------------------------------------------------

    @property
    def minimum_size(self) -> tuple[int, int]:
        """
        The smallest size the whole panel area is usable at.

        The columns sit side by side, so their minimum widths add up while the
        height is whichever column needs more of it. Only a real split costs the
        extra pixels of its sash.
        """
        docks = [self._docks[placement] for placement, _ in self._layout.docks()]
        if not docks:
            return (0, 0)

        width = sum(dock.minimum_size[0] for dock in docks)
        if len(docks) > 1:
            width += SASH

        return width, max(dock.minimum_size[1] for dock in docks)

    def _apply_window_minimum(self) -> None:
        """
        Stop the window being shrunk below what the docked panels need.

        The bar falls again when a roomier layout is applied, so a demanding
        layout does not permanently raise it.
        """
        minimum_width, minimum_height = self.minimum_size
        base_width, base_height = self._window_minimum
        chrome_width, chrome_height = self._window_chrome or (0, 0)

        self.winfo_toplevel().wm_minsize(
            max(minimum_width + chrome_width, base_width, MIN_WINDOW_SIZE[0]),
            max(minimum_height + chrome_height, base_height, MIN_WINDOW_SIZE[1]),
        )

    def _on_configure(self, _event: tk.Event) -> None:
        """
        Measure the window chrome once the window has been laid out.

        This is the first chance to measure the room the window keeps for its
        own menu bar, which the minimum size needs and which cannot be known
        before the window has been laid out.
        """
        if self._window_chrome is None:
            self._window_chrome = self._read_window_chrome()
            self._apply_window_minimum()

    def _clamp_sash(self, _event: tk.Event | None = None) -> None:
        """
        Pull the sash back if a drag would push a column below its minimum.

        This runs on every button motion inside the panels, so it leaves the
        sash alone unless it is actually out of bounds.
        """
        docks = [self._docks[placement] for placement, _ in self._layout.docks()]
        if len(docks) < 2:
            return

        total = self._panes.winfo_width()
        if total <= 1:
            return

        try:
            position = self._panes.sashpos(0)
        except tk.TclError:
            return

        low = docks[0].minimum_size[0]
        high = total - docks[-1].minimum_size[0] - SASH
        clamped = max(low, min(high, position))

        try:
            if position != clamped:
                self._panes.sashpos(0, clamped)
        except tk.TclError:
            pass
