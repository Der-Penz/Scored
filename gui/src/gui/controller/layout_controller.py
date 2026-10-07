from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.events.event_types import PanelClosed
from gui.model.layout import Layout, Panel, Placement, save_layout
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView

VisibilityListener = Callable[[bool], None]


class LayoutController(BaseController):
    """
    Owns the View menu and keeps the panel layout in sync with the model.

    Every panel gets a submenu of radio buttons, one per :class:`Placement`,
    because a panel can only be in one place at a time. A choice turns into a
    new immutable :class:`Layout`, which is handed to the
    :class:`~gui.view.panel_manager.PanelManager` and then persisted. Listeners
    registered per panel are told whether that panel ended up on screen, which
    is how the camera feed stops rendering while hidden.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._listeners: dict[Panel, list[VisibilityListener]] = {
            panel: [] for panel in Panel
        }
        self._vars: dict[Panel, tk.StringVar] = {}

    # -- setup ------------------------------------------------------------

    def add_visibility_listener(
        self, panel: Panel, listener: VisibilityListener
    ) -> None:
        """
        Call *listener* whenever *panel* becomes visible or hidden.

        The listener is called immediately with the current state so callers do
        not have to seed it themselves.
        """
        self._listeners[panel].append(listener)
        listener(self._view.layout.is_visible(panel))

    def bind_menu(self, menu: ttk.Menu) -> None:
        """Build the View menu, one submenu per panel."""
        for panel in Panel:
            var = tk.StringVar(value=self._view.layout.placement_of(panel).value)
            self._vars[panel] = var

            submenu = type(menu)(menu, tearoff=False)
            for placement in Placement:
                submenu.add_radiobutton(
                    label=placement.label,
                    value=placement.value,
                    variable=var,
                    command=lambda p=panel: self._on_placement_chosen(p),
                )
            menu.add_cascade(label=panel.label, menu=submenu)

    def bind_components(self) -> None:
        self._event_channel.subscribe(PanelClosed, self._on_panel_closed)

    def start(self) -> None:
        """Sync the menu and the listeners with the layout already on screen."""
        self._sync(self._view.layout)

    # -- menu actions -----------------------------------------------------

    def _on_placement_chosen(self, panel: Panel) -> None:
        """Move a panel to the placement picked from its submenu."""
        placement = Placement(self._vars[panel].get())
        self._commit(self._view.layout.with_placement(panel, placement))

    def _on_panel_closed(self, event: PanelClosed) -> None:
        """Hide a panel whose floating window the user closed."""
        self._commit(self._view.layout.with_placement(event.panel, Placement.HIDDEN))

    # -- plumbing ---------------------------------------------------------

    def _commit(self, layout: Layout) -> None:
        """Apply, persist and announce a new layout."""
        self._view.apply_layout(layout)
        save_layout(layout)
        self._sync(layout)
        logging.debug(
            "Layout: "
            + ", ".join(
                f"{panel.label} {layout.placement_of(panel).label}" for panel in Panel
            )
        )

    def _sync(self, layout: Layout) -> None:
        """Push the new state back into the menu and the listeners."""
        for panel in Panel:
            var = self._vars.get(panel)
            if var is not None:
                var.set(layout.placement_of(panel).value)
            for listener in self._listeners[panel]:
                listener(layout.is_visible(panel))
