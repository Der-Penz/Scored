"""Controller for the View menu: which panels are shown and where they sit."""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.model.layout import (
    Column,
    Layout,
    Panel,
    column_key,
    column_options,
    default_layout,
    save_layout,
)
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView

VisibilityListener = Callable[[bool], None]


class LayoutController(BaseController):
    """
    Owns the View menu and keeps the panel layout in sync with the model.

    Every menu action turns into a new immutable :class:`Layout`, which is handed
    to the :class:`~gui.view.panel_manager.PanelManager` and then persisted.
    Listeners registered per panel are told whether that panel ended up on
    screen, which is how the camera feed stops rendering while hidden.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._listeners: dict[Panel, list[VisibilityListener]] = {panel: [] for panel in Panel}
        self._column_vars: dict[Column, tk.StringVar] = {}
        self._visible_vars: dict[Panel, tk.BooleanVar] = {}

    # -- setup ------------------------------------------------------------

    def add_visibility_listener(self, panel: Panel, listener: VisibilityListener) -> None:
        """
        Call *listener* whenever *panel* becomes visible or hidden.

        The listener is called immediately with the current state so callers do
        not have to seed it themselves.
        """
        self._listeners[panel].append(listener)
        listener(self._view.layout.is_visible(panel))

    def bind_menu(self, menu: tk.Menu) -> None:
        """Build the View menu from the panels and the layout."""
        for panel in Panel:
            var = tk.BooleanVar(value=self._view.layout.is_visible(panel))
            self._visible_vars[panel] = var
            menu.add_checkbutton(
                label=f"Show {panel.label}",
                variable=var,
                command=lambda p=panel: self._on_toggle(p),
            )

        menu.add_separator()

        for column in Column:
            var = tk.StringVar(value=column_key(self._view.layout.column(column)))
            self._column_vars[column] = var
            submenu = type(menu)(menu, tearoff=False)
            for key, label, _panels in column_options():
                submenu.add_radiobutton(
                    label=label,
                    value=key,
                    variable=var,
                    command=lambda c=column: self._on_column_chosen(c),
                )
            menu.add_cascade(label=column.label, menu=submenu)

        menu.add_separator()
        menu.add_command(label="Swap Columns", command=self._on_swap)
        menu.add_command(label="Reset Layout", command=self._on_reset)

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        """Sync the menu and the listeners with the layout already on screen."""
        self._sync_widgets(self._view.layout)

    # -- menu actions -----------------------------------------------------

    def _on_toggle(self, panel: Panel) -> None:
        """Flip the visibility of a panel from its Show checkbutton."""
        layout = self._view.layout

        if layout.is_visible(panel):
            self._commit(layout.with_panel(panel, None))
            return

        self._commit(layout.with_panel(panel, layout.last_column.get(panel) or Column.LEFT))

    def _on_column_chosen(self, column: Column) -> None:
        """Replace the contents of a column from its submenu selection."""
        key = self._column_vars[column].get()
        for option_key, _label, panels in column_options():
            if option_key == key:
                self._commit(self._view.layout.with_column(column, panels or ()))
                return

    def _on_swap(self) -> None:
        """Exchange the left and right columns."""
        self._commit(self._view.layout.swap_columns())

    def _on_reset(self) -> None:
        """Return to the default layout."""
        self._commit(default_layout())

    # -- plumbing ---------------------------------------------------------

    def _commit(self, layout: Layout) -> None:
        """Apply, persist and announce a new layout."""
        self._view.apply_layout(layout)
        save_layout(layout)
        self._sync_widgets(layout)
        logging.info(
            "Layout: "
            + ", ".join(
                f"{column.label} [{', '.join(p.label for p in panels) or 'empty'}]"
                for column, panels in layout.columns()
            )
        )

    def _sync_widgets(self, layout: Layout) -> None:
        """Push the new state back into the menu and the listeners."""
        for panel in Panel:
            var = self._visible_vars.get(panel)
            if var is not None:
                var.set(layout.is_visible(panel))
            for listener in self._listeners[panel]:
                listener(layout.is_visible(panel))

        for column in Column:
            var = self._column_vars.get(column)
            if var is not None:
                var.set(column_key(layout.column(column)))
